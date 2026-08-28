# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL sparse MLA prefill kernel, DSv4 two-region only (gfx942 / CDNA3).

A specialization of ``sparse_mla_prefill.py`` carrying exactly one code path:
the DeepSeek-v4 two-region packed ``fp8_ds_mla`` prefill.  Everything the
general kernel supports and DSv4 never takes -- the Phase A flat builder, the
GLM/DSv3.2 ``glm_flat576`` layout, single-region loops, ``qk_split=False``,
``block_n != 32``, ``scale_mode='per_tensor'`` -- is gone, so the tile loop can
be pipelined without the branch scaffolding those paths require.

Fixed at compile time:
  head_dim = 512 (448 fp8 NoPE + 64 bf16 RoPE), v_dim = 512, num_regions = 2,
  block_n = 32, packed fp8_ds_mla cache, one CTA per query, 8 warps.

Still selectable (all ``const_expr``, one compiled kernel per combination):
  has_sink, r0_convert (region0 UE8M0 scales), r0_is_ocp / r1_is_ocp (per-region
  fp8 convention), single_request (q_req indirection), rope_bf16.

Region roles are positional and inherited from the general kernel: ``main_*`` is
region0 and must be non-empty for every query (it seeds the shared online-softmax
state and sets the per-query tile count); ``extra_*`` is region1 and may be
empty.  Tile space is region0 tiles [0, n0) then region1 tiles [0, n1).

gfx942 only -- no gfx950 intrinsics (no ds_read_b64_tr_b8, no wide DMA).

NOTE: Do NOT use ``from __future__ import annotations`` here -- it breaks
``fx.Constexpr`` detection in the FlyDSL AST rewriter.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator

# ---------------------------------------------------------------------------
# Compile-time constants (DSv4: head_dim=512, v_dim=512, block_n=32)
# ---------------------------------------------------------------------------
NUM_QO_HEADS: int = 128
KV_LORA_RANK: int = 512
QK_HEAD_DIM: int = 512
V_HEAD_DIM: int = 512
NUM_WARPS: int = 8
WARP_SIZE: int = 64
NUM_THREADS: int = NUM_WARPS * WARP_SIZE  # 512
BLOCK_M: int = 128  # == NUM_QO_HEADS
BLOCK_N: int = 32
BLOCK_K: int = 32
TILE_M: int = BLOCK_M // NUM_WARPS  # 16
LOG2E: float = 1.4426950408889634
NEG_LARGE: float = -3.4028234663852886e38

# ---- KvManagerV2 LDS layout (32 rows x 512 cols fp8, 8 blocks of 64 cols) ----
KV_NUM_COLS: int = 64
KV_NUM_BLOCKS: int = QK_HEAD_DIM // KV_NUM_COLS  # 8
KV_ROWS_PER_SUB: int = BLOCK_N // NUM_WARPS  # 4
KV_BYTES_PER_ROW: int = KV_NUM_COLS  # 64 (fp8)
KV_PAD_DW: int = 2
KV_SUB_BYTES: int = KV_ROWS_PER_SUB * KV_BYTES_PER_ROW + KV_PAD_DW * 4  # 264
KV_NUM_SUBS: int = BLOCK_N // KV_ROWS_PER_SUB  # 8
KV_BLOCK_BYTES: int = KV_SUB_BYTES * KV_NUM_SUBS  # 2112
SZ_LDS_KV: int = KV_BLOCK_BYTES * KV_NUM_BLOCKS  # 16896

# ---- VtManagerV1 LDS layout (software-transposed V staging; vt_inreg=False) --
VT_ROWS_PER_THR: int = 4
VT_COLS_PER_THR: int = 8
VT_ELEMS_PER_BLK: int = VT_ROWS_PER_THR * VT_COLS_PER_THR  # 32
VT_BLKS_PER_ROW: int = V_HEAD_DIM // VT_COLS_PER_THR  # 64
VT_BLKS_PER_ROW_PAD: int = VT_BLKS_PER_ROW + 2  # 66
VT_NUM_SUB_BLKS: int = 8
SZ_LDS_VT: int = VT_NUM_SUB_BLKS * ((BLOCK_N // VT_NUM_SUB_BLKS) * V_HEAD_DIM + 16 * 4)  # 16896
# De-interleaved byte layout: each lane's 4x8 fp8 block splits into two 16-B
# halves with a 16-B col-block stride, so an 8-lane ds_write_b128 phase tiles all
# 32 banks. Pure byte permutation; the reader matches.
VT_ROWBLK_STRIDE: int = VT_BLKS_PER_ROW_PAD * VT_ELEMS_PER_BLK  # 2112
VT_HALF_STRIDE: int = VT_BLKS_PER_ROW_PAD * 16  # 1056
VT_COLBLK_STRIDE: int = 16
VT_OFFSET_TL_BL: int = 4 * VT_ROWBLK_STRIDE  # 8448

# ---- VtManagerWide: operand-major V^T staging (vt_wide=True) ---------------
# The default layout gives a warp a 16-row x 128-col quadrant, so a lane holds
# only 4 of the 8 rows an A operand needs and GEMM2 must reassemble each operand
# from two ds_read_b32 a VT_OFFSET_TL_BL apart. Splitting V as 32 rows x 64 cols
# per warp instead lets a lane read 8 rows x 4 cols, transpose to 4 *complete*
# operands, and store them as 32 contiguous bytes -- so the write stays 2 x
# ds_write_b128 and GEMM2 reads one ds_read_b64 per operand. Net -28 LDS
# instructions per lane per tile, which matters because this kernel is
# instruction-issue bound.
#
#     addr(q, col) = q*VT2_Q_STRIDE + (col//8)*VT2_GRP + (col%8)*8
#
# VT2_GRP must be a multiple of 16 so the ds_write_b128 stays 16-byte aligned
# and the ds_read_b64 8-byte aligned -- a padded stride breaks both and the
# compiler splits them (measured 2x slower). Unpadded (64) is conflict-free for
# the 32 reads per lane per tile and only 2-way for the 2 writes, which is the
# right side to favour.
VT2_GRP: int = 8 * 8  # 64
VT2_Q_STRIDE: int = (V_HEAD_DIM // 8) * VT2_GRP  # 4352
SZ_LDS_VT2: int = 4 * VT2_Q_STRIDE  # 17408

# ---- OManager16bitsV2 (bf16 output via LDS reshape; reuses KV buffer 0) ----
O16_NUM_ROWS: int = 16
O16_NUM_COLS: int = 32
O16_PAD_ELEM_PER_2ROWS: int = 4
O16_ELEM_PER_PAD_2ROWS: int = 2 * O16_NUM_COLS + O16_PAD_ELEM_PER_2ROWS  # 68
O16_LDS_PER_WARP: int = (O16_NUM_ROWS // 2) * O16_ELEM_PER_PAD_2ROWS * 2  # 1088
SZ_LDS_O16: int = NUM_WARPS * O16_LDS_PER_WARP  # 8704

# ---- Overall LDS layout (byte offsets) ----
# Q goes straight to VGPRs and V is transposed in registers, so neither needs a
# staging region. What remains is the double-buffered KV tile plus, under
# rope_bf16, the double-buffered bf16 RoPE K tiles. Output staging aliases KV
# buffer 0, which is dead by the time the epilogue runs.
P_LDS_KV_0: int = 0
P_LDS_KV_1: int = SZ_LDS_KV  # 16896
P_LDS_RBF: int = 2 * SZ_LDS_KV  # 33792 (only allocated when rope_bf16)

assert SZ_LDS_O16 <= SZ_LDS_KV, "Output LDS must fit in the KV buffer region"

# ---- MFMA tile constants ----
MFMA_M: int = 16
MFMA_N: int = 16
MFMA_K: int = 32  # mfma_f32_16x16x32_fp8_fp8
MFMA_ELEM_PER_THR: int = MFMA_M * MFMA_K // WARP_SIZE  # 8

NUM_NOPE_ITERS: int = QK_HEAD_DIM // (MFMA_K * 2)  # 8
NUM_PV_ITERS: int = V_HEAD_DIM // (MFMA_N * 2)  # 16
P_VALS_PER_THR: int = (BLOCK_N * MFMA_M) // WARP_SIZE  # 8

# ---- fp8_ds_mla packed-cache layout ----
# Packed uint8 cache: [num_blocks, block_size, 584].
#   token_data @ block_base + pos*576 : 448 fp8 NoPE + 128 B (64 bf16) RoPE
#   scales     @ block_base + block_size*576 + pos*8 : 7 UE8M0 bytes (+1 pad)
PK_NOPE_DIM: int = 448
PK_ROPE_DIM: int = 64
PK_TOKEN_BYTES: int = 576
PK_NOPE_BYTES: int = 448
PK_CACHE_ROW: int = 584
PK_NOPE_BLOCKS: int = PK_NOPE_DIM // KV_NUM_COLS  # 7 (cols 0..447)
PK_ROPE_BLOCK: int = PK_NOPE_BLOCKS  # block 7 holds the requant'd RoPE tail

# ---- bf16 RoPE split-dot layout (rope_bf16=True) ----
# NoPE (448, blocks 0..6) stays fp8 MFMA; the 64-d RoPE tail is dotted in bf16
# via mfma_f32_16x16x16bf16_1k accumulating into the SAME f32 p_comp[h] tile
# (identical 16x16 C lane layout -> exact merge).  The bf16 RoPE K tile is a
# shared [BLOCK_N][PK_ROPE_DIM] array kept at P_LDS_RBF.  V (GEMM2) still reads
# the fp8 rope (block 7) untouched.
RBF_KSTEP: int = MFMA_M  # 16
RBF_NUM_STEPS: int = PK_ROPE_DIM // RBF_KSTEP  # 4
RBF_ROW_PAD: int = 4  # bf16 padding per kv-row to spread LDS banks
RBF_ROW_STRIDE: int = (PK_ROPE_DIM + RBF_ROW_PAD) * 2  # 136 B per kv-row
SZ_LDS_RBF: int = BLOCK_N * RBF_ROW_STRIDE  # 4352 B per tile

def _lds_layout(rope_bf16: bool, vt_inreg: bool, vt_wide: bool = False):
    """Byte offset of the V^T region and the total allocation.

    KV double buffer first, then the (double-buffered) bf16 RoPE K tiles under
    rope_bf16, then V^T staging unless it is transposed in registers.
    """
    p_vt = P_LDS_RBF + (2 * SZ_LDS_RBF if rope_bf16 else 0)
    sz_vt = 0 if vt_inreg else (SZ_LDS_VT2 if vt_wide else SZ_LDS_VT)
    total = p_vt + sz_vt
    assert total <= 65536, f"gfx942 LDS cap: need {total} B"
    return p_vt, total


# ---- In-register V transpose addressing (vt_inreg=True) --------------------
# GEMM2's A operand wants 8 consecutive KV rows at one v-dim, i.e. V column
# major, while the KV tile is row major. Rather than transpose cooperatively
# and publish V^T through LDS (which costs a CTA barrier), every wave reads the
# whole tile and transposes what it needs in registers. That is 8x redundant
# VALU but *less* LDS traffic, because the V^T read was already 8x amplified.
#
# It only pays off if each lane's reads are contiguous, which needs a v-dim
# remap: A-tile row ``m`` of MFMA ``s`` means v-dim ``m*32 + s`` instead of
# ``s*16 + m``. Lane L then needs rows (L/16)*8..+7 -- fixed for the whole tile
# -- and columns (L%16)*32..+31, a contiguous 32-wide window.
def _v_row_phy(row: int) -> int:
    return (row % 16) // 2 * 4 + 2 * (row // 16) + row % 2


def _v_row_byte(row: int) -> int:
    p = _v_row_phy(row)
    return (p // 4) * KV_SUB_BYTES + (p % 4) * KV_BYTES_PER_ROW


# The MFMA's k index is NOT the logical KV row. GEMM1 leaves P split across two
# 16-row sub-tiles (p_comp[0] = rows 0-15, p_comp[1] = rows 16-31), so for lane
# group q the B operand's bytes are rows {4q..4q+3} then {16+4q..16+4q+3}. The A
# operand must use the same permutation -- which is exactly what the V^T layout
# encodes with VT_OFFSET_TL_BL, a jump of 4 row-blocks. So a lane reads two runs
# of 4 rows, 16 apart, not 8 consecutive rows.
VROW4_OFF = tuple(_v_row_byte(j) - _v_row_byte(0) for j in range(4))
for _r in range(0, BLOCK_N, 4):
    assert (
        tuple(_v_row_byte(_r + j) - _v_row_byte(_r) for j in range(4)) == VROW4_OFF
    ), "KV row swizzle is not 4-row-group invariant; the in-register V read needs rework"
for _q in range(4):
    assert _v_row_byte(4 * _q) == _q * (2 * KV_SUB_BYTES)
    assert _v_row_byte(16 + 4 * _q) == _q * (2 * KV_SUB_BYTES) + 2 * KV_BYTES_PER_ROW

# _transpose_v emits its 8 columns in a fixed permuted order: index i holds
# column VT_IDX_TO_COL[i]. Decoded from the c_perm0..3 selector constants.
VT_IDX_TO_COL = (0, 4, 1, 5, 2, 6, 3, 7)
VT_COL_TO_IDX = tuple(VT_IDX_TO_COL.index(c) for c in range(8))

# ---- v-dim remap for the in-register transpose ----------------------------
# Reads happen in 8-column chunks, so which chunk a lane owns is free to choose.
# The obvious choice (lane m owns the contiguous window 32m..32m+31) is the
# worst one: a chunk's byte offset is (chunk/8)*KV_BLOCK_BYTES + 8*(chunk%8),
# and KV_BLOCK_BYTES mod 128 is always a multiple of 32 for any KV_PAD_DW, so
# the 16 lanes of a phase collapse onto 4 banks -- a 4-way conflict no padding
# can fix.
#
#     chunk(m, grp) = 16*grp + 4*(m % 4) + (m // 4)
#
# fixes both problems at once:
#   * addr mod 128 becomes ((t//8)%2)*64 + 8*(t%8) with t = 4*(m%4) + m//4,
#     which is 16 distinct values over the 16 lanes -- conflict-free.
#   * accumulator 8*grp + c then holds v-dim 128*grp + 32*j + 8*(lane/16) + c
#     (j = the f32x4 element index), so for each (grp, j) the wave covers 32
#     *contiguous* v-dims -- exactly the 16x32 tile the existing epilogue
#     reshape already stages, so output stores stay coalesced.
def _v_chunk_t(m: int) -> int:
    return 4 * (m % 4) + (m // 4)


assert sorted(_v_chunk_t(m) for m in range(16)) == list(range(16))
assert (
    len({((_v_chunk_t(m) // 8) % 2) * 64 + 8 * (_v_chunk_t(m) % 8) for m in range(16)}) == 16
), "in-register V read is bank-conflicted"

# A 32-wide column window starting at a multiple of 32 never straddles a
# KV_NUM_COLS block boundary, so the 4 chunks are plain +0/+8/+16/+24.
assert KV_NUM_COLS % 32 == 0


# ---------------------------------------------------------------------------
# Module-level utility helpers
# ---------------------------------------------------------------------------
def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


def _inline_asm_void(operands, asm_string, constraints):
    llvm.inline_asm(None, operands, asm_string, constraints, has_side_effects=True)


def _barrier(vmcnt=63, lgkmcnt=63):
    parts = []
    needs_waitcnt = vmcnt < 63 or lgkmcnt < 63
    if needs_waitcnt:
        wc = []
        if vmcnt < 63:
            wc.append(f"vmcnt({vmcnt})")
        if lgkmcnt < 63:
            wc.append(f"lgkmcnt({lgkmcnt})")
        parts.append("s_waitcnt " + " ".join(wc))
    parts.append("s_barrier")
    _inline_asm_void([], "\n".join(parts), "")


def _inttoptr_lds(byte_addr):
    # NOTE: parse the result type fresh each call. Caching it module-level binds
    # the Type to the first kernel's MLIRContext; a second specialization
    # compiles in a new context and reusing the cached Type fails verification.
    lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
    return llvm.inttoptr(lds_ptr_type, _raw(fx.Int64(byte_addr)))


_gep = buffer_ops.get_element_ptr


def _ptr_load(result_type, ptr, *, alignment=None, volatile_=False, nontemporal=False):
    return llvm.LoadOp(
        result_type, ptr, alignment=alignment, volatile_=volatile_, nontemporal=nontemporal
    ).result


def _ptr_store(value, ptr, *, alignment=None, volatile_=False):
    return llvm.StoreOp(_raw(value), ptr, alignment=alignment, volatile_=volatile_)


def _lds_load(byte_addr_index, vec_type, static_byte_offset=0):
    lds_ptr = _inttoptr_lds(byte_addr_index)
    if static_byte_offset != 0:
        lds_ptr = _gep(lds_ptr, static_byte_offset=static_byte_offset)
    return _ptr_load(vec_type, lds_ptr, alignment=16, nontemporal=True)


def _lds_load_at(base_idx, vec_type, byte_offset=0, alignment=4):
    ptr = _inttoptr_lds(base_idx)
    if byte_offset != 0:
        ptr = _gep(ptr, static_byte_offset=byte_offset)
    return _ptr_load(vec_type, ptr, alignment=alignment, nontemporal=True)


def _lds_load_volatile(base_i32, vec_type, byte_offset=0):
    lds_ptr = _inttoptr_lds(ArithValue(base_i32).extui(T.i64))
    if byte_offset != 0:
        lds_ptr = _gep(lds_ptr, static_byte_offset=byte_offset)
    return _ptr_load(vec_type, lds_ptr, alignment=8, volatile_=True)


def _lds_ptr_from_i32(addr_i32, byte_offset=0):
    ptr = _inttoptr_lds(ArithValue(addr_i32).extui(T.i64))
    if byte_offset != 0:
        ptr = _gep(ptr, static_byte_offset=byte_offset)
    return ptr


def _i32(value):
    raw = _raw(value) if not isinstance(value, ir.Value) else value
    if raw.type == T.i32:
        return raw
    return _raw(fx.Int32(raw))


def _uniform_i32(value):
    return rocdl.readfirstlane(T.i32, _i32(value))


def _fast_exp2(val):
    return rocdl.exp2(T.f32, _raw(val))


def _f32(val):
    if isinstance(val, fx.Float32):
        return val
    if isinstance(val, (int, float)):
        return fx.Float32(float(val))
    return fx.Float32(val)


def _idx(val):
    if isinstance(val, fx.Index):
        return val
    return fx.Index(val)


def _pack_i32x2(lo, hi):
    return _raw(ArithValue(lo).extui(T.i64) | (ArithValue(hi).extui(T.i64) << 32))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def compile_sparse_mla_prefill_dsv4(
    *,
    has_sink: bool = False,
    r0_convert: bool = False,
    r0_is_ocp: bool = False,
    r1_is_ocp: bool = True,
    r1_convert: bool = True,
    softmax_scale: float | None = None,
    single_request: bool = True,
    rope_bf16: bool = False,
    rope_fp8: bool = False,
    r1_tb_carry: bool = False,
    vt_inreg: bool = False,
    kv_double_buffer: bool = False,
    kv_pf_late: bool = False,
    vt_wide: bool = False,
    rope_prefetch: bool = True,
    scale_coalesce: bool = True,
    slot_hoist: bool = False,
    xcd_remap: bool = True,
    xcd_count: int = 8,
):
    """Build the DSv4 two-region paged sparse MLA prefill launcher (gfx942).

    has_sink:    fold a per-head virtual key into the softmax denominator.
    r0_convert:  region0 uses the register-staged convert load (needed when
                 UE8M0 != 1 or region0 is OCP); else the fast DMA path.
    r1_convert:  region1 uses the register-staged convert load (dequant by the
                 UE8M0 exponent, requantize, ds_write). False routes region1
                 through region0's fire-and-forget DMA instead, which requires the
                 extra cache to already be fnuz with unity scale -- i.e. converted
                 by a preprocessing pass. r1 tiles cost ~1.8x r0 tiles today and
                 the convert is why, so this is where region1's headroom is.
    r0_is_ocp / r1_is_ocp: per-region NoPE fp8 convention (fnuz vs OCP).  The
                 fnuz/OCP correction only exists inside the convert load, so a
                 DMA'd region0 must already be fnuz (r0_convert=False implies
                 r0_is_ocp=False).
    single_request: hardwire req_id=0 instead of reading q_req[q].
    rope_fp8:    the cache's RoPE tail is already fnuz fp8 in the 64 bytes at
                 PK_NOPE_BYTES, so block 7 rides the same buffer_load_to_lds as
                 blocks 0..6 and the bf16->fp8 convert disappears. That convert's
                 s_waitcnt was the largest single stall in the kernel (571 + 160
                 cyc/wave-tile across the two regions). Mutually exclusive with
                 rope_bf16, which needs the bf16 bytes the fp8 overwrites.
    r1_tb_carry: carry region1's token base across its tiles the way TB_CARRY does
                 for region0, so its KV DMA can issue at the top of the tile instead
                 of after the address chain. ATT motivated it -- region1 pays 615
                 cyc/tile at the tile barrier against region0's 169, because the DMA
                 cannot go out until the chain lands.

                 MEASURED +1.21% [+1.10, +1.31] SLOWER, and it is not the boundary
                 branch: the first version resolved the r0->r1 boundary tile with a
                 dynamic if and cost +1.37%; hoisting that tile's base to the
                 prologue (it is a per-query constant) and selecting with a
                 v_cndmask recovered only 0.16%.

                 The real blocker is in-order vmcnt. Issuing the DMA before the
                 chain gains nothing because _row_addrs's own two dependent
                 vmcnt(0) drains immediately flush it -- and it is now older in the
                 queue, so it is what the drain waits on first. Reordering cannot
                 help here; only partial counts or one less dependent load would.
                 Off by default; kept for A/B.
    rope_bf16:   dot the 64 RoPE dims in bf16 (vLLM NoPE-fp8 / RoPE-bf16
                 contract) instead of re-quantizing them to fp8.
    vt_inreg:    read V straight from the KV tile in GEMM2 and transpose it in
                 registers instead of publishing V^T through LDS. Removes a
                 barrier and SZ_LDS_VT, but costs +112 v_perm per lane per tile,
                 which measured net slower on gfx942 -- the kernel is issue
                 bound, and there is no hardware byte-transpose (that is gfx950's
                 ds_read_b64_tr_b8). Off by default; kept for A/B.
    kv_double_buffer: DMA the next tile's NoPE blocks into the alternate KV
                 buffer during this tile's QK MFMAs. Cuts stall ~8%, but adds 7
                 inline m0/DMA sequences per tile and measured net slower -- this
                 kernel is instruction-issue bound, not latency bound. Off by
                 default; kept for A/B.
    kv_pf_late:  with kv_double_buffer, issue the 7 prefetch DMAs in one batch
                 after the softmax instead of interleaved into the QK MFMA
                 loop. GEMM1 is the LDS-busiest phase, so the DMA writes
                 contend there; issuing after leaves GEMM2 (~2000 cycles) to
                 land, and keeps them clear of the softmax's slot loads in
                 the in-order vmcnt.
    vt_wide:     split V as 32 rows x 64 cols per warp so a lane transposes
                 complete 8-row A operands. GEMM2 then reads one ds_read_b64 per
                 operand instead of two ds_read_b32, cutting 32 LDS instructions
                 per lane per tile for 4 more on the V read. Ignored under
                 vt_inreg (which has no V^T at all).
    rope_prefetch: issue the RoPE tail's global load one tile ahead and commit
                 it (cvt + ds_write) at the top of the next tile. Unlike the
                 NoPE DMA this load is register-staged, so its s_waitcnt sits
                 at the top of the tile and -- vmcnt being in-order -- drags
                 the NoPE DMAs with it. Only pays off together with
                 kv_double_buffer -- but MEASURED: rope_prefetch alone is the
                 best of the four corners (-2.2% vs neither), because
                 kv_double_buffer's own 7 m0/DMA sequences per tile cost more
                 than the DMA latency they hide. On by default.
    scale_coalesce: fetch a token's UE8M0 exponents with one dwordx2 hoisted out
                 of the block loop instead of one dword per block. The per-block
                 form costs 7 fetches the compiler will not CSE even though
                 several share an address. Removes 6 loads and 23 instructions,
                 cut total vmcnt drain 7% in ATT, and MEASURED -0.39% [-0.27,
                 -0.57] on the paired harness. On by default.
    slot_hoist:  issue tile 0's CSR slot load ahead of the Q loads so the two
                 round trips overlap, taking the prologue chain from four serial
                 round trips to three. ATT agreed (-124 cyc/wave-tile on the
                 prologue) but wall time MEASURED +0.60% [+0.44, +0.67] SLOWER on
                 the paired harness, with VGPR count unchanged -- another instance
                 of exposed per-wave latency here not being the thing that costs.
                 Off by default; kept for A/B.
    xcd_remap:   block the workgroup->XCD mapping instead of the hardware's
                 round-robin, so one XCD owns a contiguous run of queries and
                 streams contiguous Q / output / CSR ranges. MEASURED -1.45% on
                 i.i.d. indices and -2.60% with realistic locality; see the note at
                 the remap for why most of it is stream contiguity rather than
                 cross-query KV reuse. On by default.
    xcd_count:   XCDs to block across (8 for MI300X in SPX; wrong under CPX/NPS).
    """
    HAS_SINK = bool(has_sink)
    R0_CONVERT = bool(r0_convert)
    R0_OCP = bool(r0_is_ocp)
    R1_OCP = bool(r1_is_ocp)
    R1_CONVERT = bool(r1_convert)
    SINGLE_REQUEST = bool(single_request)
    ROPE_BF16 = bool(rope_bf16)
    ROPE_FP8 = bool(rope_fp8)
    VT_INREG = bool(vt_inreg)
    KV_DB = bool(kv_double_buffer)
    KV_PF_LATE = bool(kv_pf_late)
    VT_WIDE = bool(vt_wide) and not bool(vt_inreg)
    ROPE_PF = bool(rope_prefetch)
    SCALE_COALESCE = bool(scale_coalesce)
    SLOT_HOIST = bool(slot_hoist)
    XCD_REMAP = bool(xcd_remap)
    XCD_COUNT = int(xcd_count)
    # _row_addrs is a dependent double round-trip (CSR index load feeds the
    # block-table load). Prefetching needs tile g+1's token base during tile
    # g, which would run that chain twice per tile -- so carry it instead:
    # tile g's tb_next IS tile g+1's tb. Only valid when region0 skips the
    # convert path, which is the only consumer of the scale base.
    # ROPE_FP8 retires the rope prefetch, but the token-base carry is worth keeping
    # on its own: it moves _row_addrs a tile earlier rather than adding a call.
    TB_CARRY = (KV_DB or ROPE_PF or ROPE_FP8) and not R0_CONVERT
    # Region1 can carry too, but only off the convert path: the convert is the sole
    # consumer of the scale base, and only the token base is carried.
    R1_TB_CARRY = TB_CARRY and not R1_CONVERT and bool(r1_tb_carry)
    P_LDS_VT, TOTAL_LDS_BYTES = _lds_layout(ROPE_BF16, VT_INREG, VT_WIDE)

    if ROPE_FP8 and ROPE_BF16:
        raise ValueError(
            "rope_fp8=True consumes the bf16 rope tail's storage, so it cannot be "
            "combined with rope_bf16=True, which needs those bytes for the QK dot"
        )
    if R0_OCP and not R0_CONVERT:
        raise ValueError(
            "r0_is_ocp=True requires r0_convert=True: the OCP->fnuz exponent "
            "correction only exists on the convert load path"
        )

    # With rope_bf16 the last fp8 QK iteration (block 7, the requantized rope
    # tail) is replaced by RBF_NUM_STEPS bf16 MFMA steps.
    NUM_FP8_QK_ITERS = (NUM_NOPE_ITERS - 1) if ROPE_BF16 else NUM_NOPE_ITERS

    base_scale = (QK_HEAD_DIM ** -0.5) if softmax_scale is None else float(softmax_scale)
    SOFTMAX_SCALE = fx.Float32(base_scale)

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def kn_sparse_mla_prefill_dsv4(
        query: fx.Tensor,            # [nq*128, 512] bf16
        main_cache: fx.Tensor,       # uint8 packed fp8_ds_mla (region0)
        main_indices: fx.Tensor,     # i32 CSR values (slot ids)
        main_indptr: fx.Tensor,      # i32 [nq+1]
        main_block_table: fx.Tensor, # i32 [num_reqs*max_blocks]
        extra_cache: fx.Tensor,      # uint8 packed fp8_ds_mla (region1)
        extra_indices: fx.Tensor,
        extra_indptr: fx.Tensor,
        extra_block_table: fx.Tensor,
        q_req: fx.Tensor,            # i32 [nq] query -> request id (ignored if SINGLE_REQUEST)
        sink_buf: fx.Tensor,         # f32 [128] (ignored if not HAS_SINK)
        final_output: fx.Tensor,     # [nq*128, 512] bf16
        softmax_scale: fx.Float32,
        main_num_rows: fx.Int32,
        extra_num_rows: fx.Int32,
        main_block_size: fx.Int32,
        extra_block_size: fx.Int32,
        main_max_blocks: fx.Int32,
        extra_max_blocks: fx.Int32,
    ):
        fm_no_inf = (
            arith.FastMathFlags.nnan
            | arith.FastMathFlags.nsz
            | arith.FastMathFlags.arcp
            | arith.FastMathFlags.contract
            | arith.FastMathFlags.afn
            | arith.FastMathFlags.reassoc
        )

        def _mfma_fp8(result_type, operands, **kw):
            return rocdl.mfma_f32_16x16x32_fp8_fp8(result_type, operands, **kw)

        def _mfma_bf16(c_acc, a_i16x4, b_i16x4):
            # gfx942 v_mfma_f32_16x16x16bf16_1k: A/B are vector<4xi16> (4 bf16);
            # the 16x16 f32x4 output uses the SAME lane layout as the fp8
            # 16x16x32 MFMA, so it accumulates into the shared p_comp[h] tile.
            return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a_i16x4, b_i16x4, _raw(c_acc), 0, 0, 0])

        def _bits_to_i16x4(val_i32x2):
            return _raw(Vec(Vec(val_i32x2).bitcast(fx.Int16)))

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_no_inf)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_no_inf)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_no_inf)

        def _fmax(a, b):
            return arith.maximumf(_raw(a), _raw(b), fastmath=fm_no_inf)

        arch = get_hip_arch()
        lds_allocator = SmemAllocator(None, arch=arch)
        lds_allocator.ptr = TOTAL_LDS_BYTES
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            lds_allocator.finalize()
        lds_buffer = lds_allocator.get_base()
        lds_base_idx = memref.extract_aligned_pointer_as_index(lds_buffer)

        c_perm0 = fx.Int32(0x05010400)
        c_perm1 = fx.Int32(0x07030602)
        c_perm2 = fx.Int32(0x05040100)
        c_perm3 = fx.Int32(0x07060302)

        def _vt_perm(src_hi, src_lo, sel):
            return rocdl.perm_b32(src_hi, src_lo, sel)

        c_neg_inf = fx.Float32(float("-inf"))
        c_neg_large = fx.Float32(NEG_LARGE)
        c_zero_f32 = fx.Float32(0.0)
        c_one_f32 = fx.Float32(1.0)
        c_zero_i32 = fx.Int32(0)
        c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)
        c_log2e = fx.Float32(LOG2E)

        query_rsrc = buffer_ops.create_buffer_resource(query)
        main_cache_rsrc = buffer_ops.create_buffer_resource(main_cache)
        main_indices_rsrc = buffer_ops.create_buffer_resource(main_indices)
        main_indptr_rsrc = buffer_ops.create_buffer_resource(main_indptr)
        main_bt_rsrc = buffer_ops.create_buffer_resource(main_block_table)
        extra_cache_rsrc = buffer_ops.create_buffer_resource(extra_cache)
        extra_indices_rsrc = buffer_ops.create_buffer_resource(extra_indices)
        extra_indptr_rsrc = buffer_ops.create_buffer_resource(extra_indptr)
        extra_bt_rsrc = buffer_ops.create_buffer_resource(extra_block_table)
        final_output_rsrc = buffer_ops.create_buffer_resource(final_output)
        if const_expr(not SINGLE_REQUEST):
            q_req_rsrc = buffer_ops.create_buffer_resource(q_req)
        if const_expr(HAS_SINK):
            sink_rsrc = buffer_ops.create_buffer_resource(sink_buf)
        qk_softmax_scale = softmax_scale

        if const_expr(XCD_REMAP):
            # MI300X (SPX) hands workgroup i to XCD i%8. Re-block so an XCD owns a
            # contiguous run of queries instead, which tightens the query span
            # running concurrently on one XCD from ~296 (38 CUs x stride 8) to ~37.
            #
            # MEASURED -1.45% on i.i.d. indices and -2.60% with realistic locality.
            # The larger part is NOT cross-query KV reuse: i.i.d. selections have no
            # reuse to capture at any grouping (0.8% pairwise overlap, and the same
            # per-XCD footprint either way), yet still gain 1.45%. What blocking buys
            # there is a contiguous access *stream* -- an XCD's 38 resident
            # workgroups read Q slices spanning 5.5 MB contiguous rather than 38
            # slices scattered over 42 MB, and likewise for the output write and the
            # CSR index ranges. Reuse of shared KV rows is the remaining ~1.1%, and
            # only exists when nearby queries actually select similar rows.
            xcd_n = _i32(_idx(gpu.grid_dim.x))
            xcd_pid = _i32(_idx(gpu.block_idx.x))
            xcd_per = (
                ArithValue(xcd_n) + (XCD_COUNT - 1)
            ).with_signedness(False) // XCD_COUNT
            xcd_tall = ArithValue(xcd_n).with_signedness(False) % XCD_COUNT
            xcd_tall = ArithValue(_raw(xcd_tall == 0)).select(
                _raw(fx.Int32(XCD_COUNT)), _raw(xcd_tall)
            )
            xcd_id = ArithValue(xcd_pid).with_signedness(False) % XCD_COUNT
            xcd_loc = ArithValue(xcd_pid).with_signedness(False) // XCD_COUNT
            xcd_a = ArithValue(_raw(xcd_id)) * ArithValue(_raw(xcd_per)) + ArithValue(
                _raw(xcd_loc)
            )
            xcd_b = (
                ArithValue(_raw(xcd_tall)) * ArithValue(_raw(xcd_per))
                + (ArithValue(_raw(xcd_id)) - ArithValue(_raw(xcd_tall)))
                * (ArithValue(_raw(xcd_per)) - 1)
                + ArithValue(_raw(xcd_loc))
            )
            xcd_is_tall = _raw(
                ArithValue(_raw(xcd_id)).with_signedness(False)
                < ArithValue(_raw(xcd_tall)).with_signedness(False)
            )
            q_idx = _idx(ArithValue(xcd_is_tall).select(_raw(xcd_a), _raw(xcd_b)))
        else:
            q_idx = gpu.block_idx.x
        tid = gpu.thread_id("x")
        warp_idx = tid / WARP_SIZE
        lane_idx = tid % WARP_SIZE

        if const_expr(SINGLE_REQUEST):
            req_id = c_zero_i32
        else:
            req_id = rocdl.readfirstlane(
                T.i32, buffer_ops.buffer_load(q_req_rsrc, q_idx, vec_width=1, dtype=T.i32)
            )

        kv_ld_row_base = lane_idx / 32 * 16 + (lane_idx / 16) % 2 + warp_idx * 2
        kv_ld_col_base = _i32((lane_idx % 16) * 4)

        # ---- token_base / scale_base (bytes) for this lane's KV row, or -1 ----
        def _issue_row_slot(idx_rsrc, kv_tile_start_i32):
            """Issue this lane's CSR slot load, without consuming it.

            Split out so tile 0's can be issued ahead of the Q loads: the two
            round trips then overlap under one wait instead of draining in
            series. ``phys`` cannot join them -- it depends on this value.
            """
            row_idx = kv_ld_row_base + _idx(kv_tile_start_i32)
            return buffer_ops.buffer_load(idx_rsrc, _i32(row_idx), vec_width=1, dtype=T.i32)

        def _row_addrs(
            idx_rsrc, bt_rsrc, num_rows_i32, block_size_i32, max_blocks_i32,
            kv_tile_start_i32, kv_end_i32, slot_pre=None,
        ):
            row_idx = kv_ld_row_base + _idx(kv_tile_start_i32)
            in_range = row_idx < _idx(kv_end_i32)
            if const_expr(slot_pre is None):
                slot = _issue_row_slot(idx_rsrc, kv_tile_start_i32)
            else:
                slot = slot_pre
            slot_a = ArithValue(slot)
            valid = ArithValue(_raw(in_range)) & (slot_a >= 0) & (slot_a < ArithValue(num_rows_i32))
            safe_slot = ArithValue(_raw(valid)).select(_raw(slot), _raw(c_zero_i32))
            bsz = ArithValue(block_size_i32)
            block_idx = ArithValue(safe_slot).with_signedness(False) // bsz
            pos = ArithValue(safe_slot) - ArithValue(_raw(block_idx)) * bsz
            bt_index = ArithValue(req_id) * ArithValue(max_blocks_i32) + ArithValue(_raw(block_idx))
            phys = buffer_ops.buffer_load(bt_rsrc, _i32(bt_index), vec_width=1, dtype=T.i32)
            blk_stride = bsz * PK_CACHE_ROW
            token_base = (
                ArithValue(phys) * ArithValue(_raw(blk_stride))
                + ArithValue(_raw(pos)) * PK_TOKEN_BYTES
            )
            scale_base = (
                ArithValue(phys) * ArithValue(_raw(blk_stride))
                + bsz * PK_TOKEN_BYTES
                + ArithValue(_raw(pos)) * 8
            )
            tb = ArithValue(_raw(valid)).select(_raw(token_base), _raw(fx.Int32(-1)))
            sb = ArithValue(_raw(valid)).select(_raw(scale_base), _raw(c_zero_i32))
            return _i32(tb), _i32(sb)

        # ---- NoPE DMA load (fnuz + unity-scale fast path), blocks 0..6 ----
        def _load_nope_dma(cache_rsrc, p_lds_kv_warp, token_base_i32):
            for blk in range_constexpr(PK_NOPE_BLOCKS):
                lds_adjust = blk * KV_BLOCK_BYTES - blk * KV_NUM_COLS
                lds_base_i32 = _i32(ArithValue(p_lds_kv_warp) + lds_adjust)
                is_oob = ArithValue(token_base_i32) == -1
                if is_oob:
                    lds_addr = _i32(
                        ArithValue(lds_base_i32) + blk * KV_NUM_COLS + _i32(lane_idx) * 4
                    )
                    _ptr_store(c_zero_i32, _lds_ptr_from_i32(lds_addr), alignment=4)
                else:
                    voff = _i32(ArithValue(token_base_i32) + kv_ld_col_base)
                    rocdl.buffer_load_to_lds(
                        cache_rsrc, _lds_ptr_from_i32(lds_base_i32), voff, offset=blk * KV_NUM_COLS
                    )

        # ---- fire-and-forget DMA of one NoPE block into the *next* KV buffer --
        # Same instruction as _load_nope_dma but emitted as raw asm so it can be
        # issued mid-MFMA-loop without the compiler sinking it to its use.
        def _prefetch_nope_block_asm(cache_rsrc, p_lds_kv_warp, token_base_i32, block_idx_const):
            lds_adjust = block_idx_const * KV_BLOCK_BYTES - block_idx_const * KV_NUM_COLS
            lds_base_i32 = _i32(ArithValue(p_lds_kv_warp) + lds_adjust)

            def _emit_normal_load():
                voff = _i32(ArithValue(token_base_i32) + kv_ld_col_base)
                col_off_imm = block_idx_const * KV_NUM_COLS
                lds_base_sgpr = _uniform_i32(lds_base_i32)
                asm_str = (
                    "s_mov_b32 m0, $0\n"
                    "s_nop 0\n"
                    f"buffer_load_dword $1, $2, 0 offen offset:{col_off_imm} lds"
                )
                _inline_asm_void([lds_base_sgpr, voff, _raw(cache_rsrc)], asm_str, "s,v,s")

            is_oob = ArithValue(token_base_i32) == -1
            if is_oob:
                lds_addr = _i32(
                    ArithValue(lds_base_i32) + block_idx_const * KV_NUM_COLS + _i32(lane_idx) * 4
                )
                _inline_asm_void([lds_addr, _raw(c_zero_i32)], "ds_write_b32 $0, $1", "v,v")
            else:
                _emit_normal_load()

        def _flush_nan(fval):
            bits = _raw(ArithValue(_f32(fval)).bitcast(T.i32))
            absb = ArithValue(bits) & 0x7FFFFFFF
            is_nan = _raw(absb > 0x7F800000)
            return ArithValue(is_nan).select(_raw(c_zero_f32), _raw(_f32(fval)))

        # ---- NoPE convert load (OCP and/or UE8M0), blocks 0..6 ----
        # ``bias_f32`` is the OCP->fnuz exponent correction (1.0 for OCP bytes,
        # 0.0 for fnuz) added on top of the UE8M0 (enc-127) per-block exponent.
        def _load_nope_convert(cache_rsrc, p_lds_kv_warp, token_base_i32, scale_base_i32, bias_f32):
            # A token's UE8M0 bytes are one aligned 8-byte window (PK_CACHE_ROW -
            # PK_TOKEN_BYTES == 8, both multiples of 8, and pos*8), so one dwordx2
            # holds every block's exponent. Per-block loads instead cost 7 fetches
            # that the compiler will not CSE, and each one adds a load->use edge to
            # the dequant path where the vmcnt waits land. Hoisted out of the OOB
            # arm as well: ``_row_addrs`` zeroes scale_base on an invalid row, so the
            # read stays in bounds and that arm discards it.
            if const_expr(SCALE_COALESCE):
                s_pair = Vec(
                    buffer_ops.buffer_load(
                        cache_rsrc,
                        _i32(ArithValue(scale_base_i32).with_signedness(False) // 4),
                        vec_width=2,
                        dtype=T.i32,
                    )
                )
            for blk in range_constexpr(PK_NOPE_BLOCKS):
                dst_addr = _i32(
                    ArithValue(p_lds_kv_warp) + blk * KV_BLOCK_BYTES + _i32(lane_idx) * 4
                )
                is_oob = ArithValue(token_base_i32) == -1
                if is_oob:
                    _ptr_store(c_zero_i32, _lds_ptr_from_i32(dst_addr), alignment=4)
                else:
                    byte_off = ArithValue(token_base_i32) + kv_ld_col_base + blk * KV_NUM_COLS
                    word = buffer_ops.buffer_load(
                        cache_rsrc,
                        _i32(byte_off.with_signedness(False) // 4),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    f01 = Vec(rocdl.cvt_pk_f32_fp8(T.f32x2, _raw(word), 0))
                    f23 = Vec(rocdl.cvt_pk_f32_fp8(T.f32x2, _raw(word), 1))
                    # UE8M0 per-64-block exponent byte (+ OCP x2 bias)
                    if const_expr(SCALE_COALESCE):
                        enc = (
                            ArithValue(_raw(s_pair[blk // 4])).with_signedness(False)
                            >> ((blk & 3) * 8)
                        ) & 0xFF
                    else:
                        s_byte_off = ArithValue(scale_base_i32) + blk
                        s_word = buffer_ops.buffer_load(
                            cache_rsrc,
                            _i32(s_byte_off.with_signedness(False) // 4),
                            vec_width=1,
                            dtype=T.i32,
                        )
                        s_shift = (ArithValue(s_byte_off) & 3) * 8
                        enc = (
                            ArithValue(_raw(s_word)).with_signedness(False)
                            >> ArithValue(_raw(s_shift))
                        ) & 0xFF
                    enc_f = arith.uitofp(T.f32, _raw(enc))
                    sc = _fast_exp2(_fadd(_fsub(enc_f, fx.Float32(127.0)), bias_f32))
                    f0 = _flush_nan(_fmul(f01[0], sc))
                    f1 = _flush_nan(_fmul(f01[1], sc))
                    f2 = _flush_nan(_fmul(f23[0], sc))
                    f3 = _flush_nan(_fmul(f23[1], sc))
                    w = rocdl.cvt_pk_fp8_f32(T.i32, _raw(f0), _raw(f1), c_zero_i32, 0)
                    w = rocdl.cvt_pk_fp8_f32(T.i32, _raw(f2), _raw(f3), w, 1)
                    _ptr_store(w, _lds_ptr_from_i32(dst_addr), alignment=4)

        # ---- RoPE tail (block 7): bf16 cache -> fnuz fp8 ----
        # When ROPE_BF16, the raw bf16 rope bytes are ALSO staged into the shared
        # [BLOCK_N][PK_ROPE_DIM] bf16 K tile at P_LDS_RBF for the bf16 QK dot;
        # the fp8 block 7 is still written so the V/GEMM2 path is unchanged.
        # Split into issue and commit so the load can run a tile ahead. Unlike
        # the 7 NoPE blocks this is register-staged, not a DMA -- the loaded
        # bytes are consumed by the bf16->fp8 convert -- so its s_waitcnt sits at
        # the top of the tile and, because vmcnt is in-order, drags the NoPE DMAs
        # with it. That is the chain that keeps barrier A waiting.
        def _issue_rope_load(cache_rsrc, token_base_i32):
            """Issue the RoPE tail's global load. No wait, no OOB handling.

            When token_base is -1 the computed offset is still inside the buffer
            (or clamped by the resource), so the load is safe; the commit half
            discards the value and writes zeros instead.
            """
            byte_off = ArithValue(token_base_i32) + PK_NOPE_BYTES + kv_ld_col_base * 2
            return buffer_ops.buffer_load(
                cache_rsrc,
                _i32(byte_off.with_signedness(False) // 4),
                vec_width=2,
                dtype=T.i32,
            )

        def _commit_rope_block(p_lds_kv_warp, token_base_i32, rbf_base, pair):
            dst_addr = _i32(
                ArithValue(p_lds_kv_warp) + PK_ROPE_BLOCK * KV_BLOCK_BYTES + _i32(lane_idx) * 4
            )
            is_oob = ArithValue(token_base_i32) == -1
            if const_expr(ROPE_BF16):
                rbf_addr = _i32(
                    ArithValue(_i32(rbf_base))
                    + ArithValue(_i32(kv_ld_row_base)) * RBF_ROW_STRIDE
                    + ArithValue(kv_ld_col_base) * 2
                )
            if is_oob:
                _ptr_store(c_zero_i32, _lds_ptr_from_i32(dst_addr), alignment=4)
                if const_expr(ROPE_BF16):
                    zero2 = Vec.from_elements([c_zero_i32, c_zero_i32], fx.Int32)
                    _ptr_store(zero2, _lds_ptr_from_i32(rbf_addr), alignment=8)
            else:
                if const_expr(ROPE_BF16):
                    _ptr_store(pair, _lds_ptr_from_i32(rbf_addr), alignment=8)
                bf = Vec(Vec(pair).bitcast(fx.BFloat16)).to(fx.Float32)  # 4 f32
                w = rocdl.cvt_pk_fp8_f32(T.i32, _raw(bf[0]), _raw(bf[1]), c_zero_i32, 0)
                w = rocdl.cvt_pk_fp8_f32(T.i32, _raw(bf[2]), _raw(bf[3]), w, 1)
                _ptr_store(w, _lds_ptr_from_i32(dst_addr), alignment=4)

        def _dma_rope_block(cache_rsrc, p_lds_kv_warp, token_base_i32):
            """Block 7 straight from the cache, no register round-trip.

            Valid only when the rope tail is stored pre-quantized to fnuz fp8 in the
            64 bytes at PK_NOPE_BYTES. _load_nope_dma's addressing already puts
            block 7 exactly there, so this is the same instruction blocks 0..6 use
            and it replaces the bf16->fp8 convert whose s_waitcnt was the largest
            single stall in the kernel.
            """
            blk = PK_ROPE_BLOCK
            lds_adjust = blk * KV_BLOCK_BYTES - blk * KV_NUM_COLS
            lds_base_i32 = _i32(ArithValue(p_lds_kv_warp) + lds_adjust)
            is_oob = ArithValue(token_base_i32) == -1
            if is_oob:
                lds_addr = _i32(
                    ArithValue(lds_base_i32) + blk * KV_NUM_COLS + _i32(lane_idx) * 4
                )
                _ptr_store(c_zero_i32, _lds_ptr_from_i32(lds_addr), alignment=4)
            else:
                voff = _i32(ArithValue(token_base_i32) + kv_ld_col_base)
                rocdl.buffer_load_to_lds(
                    cache_rsrc, _lds_ptr_from_i32(lds_base_i32), voff,
                    offset=blk * KV_NUM_COLS,
                )

        def _load_rope_block(cache_rsrc, p_lds_kv_warp, token_base_i32, rbf_base):
            _commit_rope_block(
                p_lds_kv_warp,
                token_base_i32,
                rbf_base,
                _issue_rope_load(cache_rsrc, token_base_i32),
            )

        # ---- K/V LDS readers (KvManagerV2 layout) ----
        k_row_in_mfma = lane_idx % MFMA_M
        k_row_phy = (k_row_in_mfma / 2) * 4 + k_row_in_mfma % 2
        k_col_in_lane = (lane_idx / MFMA_M) * MFMA_ELEM_PER_THR
        k_lds_lane_offset = (
            (k_row_phy / 4) * KV_SUB_BYTES
            + (k_row_phy % 4) * KV_BYTES_PER_ROW
            + (k_col_in_lane % KV_NUM_COLS)
        )

        def _load_k_from_lds(k_base_i32, row_offset, col_offset):
            fixed_offset = (
                (row_offset // 16) * 2 * KV_BYTES_PER_ROW
                + (col_offset % KV_NUM_COLS)
                + (col_offset // KV_NUM_COLS) * KV_BLOCK_BYTES
            )
            return _lds_load_volatile(k_base_i32, T.i64, byte_offset=fixed_offset)

        # Per-lane base for the in-register V read: lane group (lane/16) selects
        # the row quad 4q..4q+3 (its partner quad 16+4q sits a fixed 128 B away),
        # and (lane%16) selects the column chunk via the bank-spreading map
        # t = 4*(m%4) + m/4. The group's own term, 2*grp*KV_BLOCK_BYTES, is a
        # compile-time immediate folded into the read offsets.
        def _v_inreg_base(p_lds_kv_base_idx):
            q = lane_idx / 16
            lm = lane_idx % 16
            t = (lm % 4) * 4 + lm / 4
            return (
                p_lds_kv_base_idx
                + q * (2 * KV_SUB_BYTES)
                + (t / 8) * KV_BLOCK_BYTES
                + (t % 8) * 8
            )

        def _read_v_raw(v_base, grp):
            """Issue the 8 LDS reads for one 8-column chunk. No wait."""
            grp_off = 2 * grp * KV_BLOCK_BYTES
            lo = []
            hi = []
            for j in range_constexpr(4):
                d = Vec(
                    _lds_load(v_base, T.i32x2, static_byte_offset=grp_off + VROW4_OFF[j])
                )
                lo.append(d[0])
                lo.append(d[1])
            for j in range_constexpr(4):
                d = Vec(
                    _lds_load(
                        v_base,
                        T.i32x2,
                        static_byte_offset=grp_off + 2 * KV_BYTES_PER_ROW + VROW4_OFF[j],
                    )
                )
                hi.append(d[0])
                hi.append(d[1])
            return lo, hi

        # `_transpose_v` maps 4 rows x 8 cols to 8 half-operands, so calling it
        # on the low row quad and the high row quad and packing the halves gives
        # the operand the MFMA wants (low quad in bytes 0-3, high in 4-7).
        def _transpose_v_operands(raw):
            lo, hi = raw
            tl = _transpose_v(lo)
            th = _transpose_v(hi)
            return [
                _pack_i32x2(tl[VT_COL_TO_IDX[c]], th[VT_COL_TO_IDX[c]])
                for c in range_constexpr(8)
            ]

        # ---- vt_wide: 32 rows x 64 cols per warp -------------------------
        # Lane L takes operand row group q = L/16 (KV rows {4q..4q+3} and
        # {16+4q..16+4q+3}, the pairing the P operand already uses) and the 4
        # columns 64*warp + 4*(L%16). Eight ds_read_b32, one per row.
        def _load_v_rows8(p_lds_kv_base_idx, warp_idx_val, lane_idx_val):
            q = lane_idx_val / 16
            base = (
                p_lds_kv_base_idx
                + warp_idx_val * KV_BLOCK_BYTES
                + (lane_idx_val % 16) * 4
                + q * (2 * KV_SUB_BYTES)
            )
            rows = []
            for j in range_constexpr(4):
                rows.append(_lds_load_at(base, T.i32, byte_offset=VROW4_OFF[j]))
            for j in range_constexpr(4):
                rows.append(
                    _lds_load_at(
                        base, T.i32, byte_offset=2 * KV_BYTES_PER_ROW + VROW4_OFF[j]
                    )
                )
            return rows

        # 8 dwords (8 rows x 4 cols) -> 4 complete operands, emitted as
        # [lo0, hi0, lo1, hi1, ...] so operand c occupies bytes 8c..8c+7.
        # Reuses the same four v_perm selectors as the 4x8 transpose, 16 total.
        def _transpose_v8(rows8):
            def _quad(d0, d1, d2, d3):
                t0 = _vt_perm(d1, d0, c_perm0)
                t2 = _vt_perm(d1, d0, c_perm1)
                t1 = _vt_perm(d3, d2, c_perm0)
                t3 = _vt_perm(d3, d2, c_perm1)
                return [
                    _vt_perm(t1, t0, c_perm2),
                    _vt_perm(t1, t0, c_perm3),
                    _vt_perm(t3, t2, c_perm2),
                    _vt_perm(t3, t2, c_perm3),
                ]

            lo = _quad(rows8[0], rows8[1], rows8[2], rows8[3])
            hi = _quad(rows8[4], rows8[5], rows8[6], rows8[7])
            out = []
            for c in range_constexpr(4):
                out.append(lo[c])
                out.append(hi[c])
            return out

        def _store_vt8_to_lds(vt_region_idx, warp_idx_val, lane_idx_val, ops8):
            q = lane_idx_val / 16
            lm = lane_idx_val % 16
            addr = (
                vt_region_idx
                + q * VT2_Q_STRIDE
                + (warp_idx_val * 8 + lm / 2) * VT2_GRP
                + (lm % 2) * 32
            )
            Vec(Vec.from_elements(ops8[0:4], fx.Int32)).bitcast(fx.Int8).store(
                lds_buffer, [addr]
            )
            Vec(Vec.from_elements(ops8[4:8], fx.Int32)).bitcast(fx.Int8).store(
                lds_buffer, [addr + 16]
            )

        def _load_vt8_from_lds(vt_base_i32, col_offset):
            return _lds_load_volatile(
                vt_base_i32, T.i64, byte_offset=(col_offset // 8) * VT2_GRP
            )

        def _vt8_base_i32():
            lm = lane_idx % 16
            off = (lane_idx / 16) * VT2_Q_STRIDE + (lm / 8) * VT2_GRP + (lm % 8) * 8
            return _i32(ArithValue(lds_base_idx + P_LDS_VT) + off)

        def _load_v_from_lds(p_lds_kv_base_idx, warp_idx_val, lane_idx_val):
            row = (warp_idx_val % 2) * 16 + (lane_idx_val / 16) * 4
            row_mod16 = row % 16
            row_phy = (row_mod16 / 2) * 4 + 2 * (row / 16) + row % 2
            col = (lane_idx_val % 16) * 8 + (warp_idx_val / 2) * 128
            lds_v_offset = (
                (row_phy / 4) * KV_SUB_BYTES
                + (row_phy % 4) * KV_BYTES_PER_ROW
                + (col / KV_NUM_COLS) * KV_BLOCK_BYTES
                + (col % KV_NUM_COLS)
            )
            lds_addr = p_lds_kv_base_idx + lds_v_offset
            v_vals = []
            for pass_idx in range_constexpr(4):
                if const_expr(pass_idx == 0):
                    off = 0
                elif const_expr(pass_idx == 1):
                    off = KV_BYTES_PER_ROW
                elif const_expr(pass_idx == 2):
                    off = KV_SUB_BYTES
                else:
                    off = KV_SUB_BYTES + KV_BYTES_PER_ROW
                data = Vec(_lds_load(lds_addr, T.i32x2, static_byte_offset=off))
                v_vals.append(data[0])
                v_vals.append(data[1])
            return v_vals

        def _store_vt_to_lds(vt_lds_base_idx, warp_idx_val, lane_idx_val, vt8):
            row_blk = (warp_idx_val % 2) * 4 + lane_idx_val / 16
            col_blk = (lane_idx_val % 16) + (warp_idx_val / 2) * 16
            lo_addr = vt_lds_base_idx + row_blk * VT_ROWBLK_STRIDE + col_blk * VT_COLBLK_STRIDE
            hi_addr = lo_addr + VT_HALF_STRIDE
            Vec(Vec.from_elements(vt8[0:4], fx.Int32)).bitcast(fx.Int8).store(
                lds_buffer, [lo_addr]
            )
            Vec(Vec.from_elements(vt8[4:8], fx.Int32)).bitcast(fx.Int8).store(
                lds_buffer, [hi_addr]
            )

        def _load_vt_from_lds(vt_base_i32, col_offset):
            blk = (col_offset // VT_COLS_PER_THR) * VT_COLBLK_STRIDE
            v0 = _lds_load_volatile(vt_base_i32, T.i32, byte_offset=blk)
            v1 = _lds_load_volatile(vt_base_i32, T.i32, byte_offset=blk + VT_OFFSET_TL_BL)
            return v0, v1

        def _vt_base_i32():
            vt_row_blk = lane_idx / 16
            vt_col_blk = (lane_idx % 16) / VT_COLS_PER_THR
            vt_row_inblk = lane_idx % VT_ROWS_PER_THR
            vt_col_inblk = ((lane_idx % 8) / VT_ROWS_PER_THR) * VT_ROWS_PER_THR
            off = (
                vt_row_blk * VT_ROWBLK_STRIDE
                + (vt_row_inblk / 2) * VT_HALF_STRIDE
                + vt_col_blk * VT_COLBLK_STRIDE
                + (vt_row_inblk % 2) * VT_COLS_PER_THR
                + vt_col_inblk
            )
            return _i32(ArithValue(lds_base_idx + P_LDS_VT) + off)

        def _transpose_v(v8):
            t0_0 = _vt_perm(v8[2], v8[0], c_perm0)
            t2_0 = _vt_perm(v8[2], v8[0], c_perm1)
            t0_1 = _vt_perm(v8[3], v8[1], c_perm0)
            t2_1 = _vt_perm(v8[3], v8[1], c_perm1)
            t1_0 = _vt_perm(v8[6], v8[4], c_perm0)
            t3_0 = _vt_perm(v8[6], v8[4], c_perm1)
            t1_1 = _vt_perm(v8[7], v8[5], c_perm0)
            t3_1 = _vt_perm(v8[7], v8[5], c_perm1)
            r = [None] * 8
            r[0] = _vt_perm(t1_0, t0_0, c_perm2)
            r[1] = _vt_perm(t1_1, t0_1, c_perm2)
            r[2] = _vt_perm(t1_0, t0_0, c_perm3)
            r[3] = _vt_perm(t1_1, t0_1, c_perm3)
            r[4] = _vt_perm(t3_0, t2_0, c_perm2)
            r[5] = _vt_perm(t3_1, t2_1, c_perm2)
            r[6] = _vt_perm(t3_0, t2_0, c_perm3)
            r[7] = _vt_perm(t3_1, t2_1, c_perm3)
            return r

        def _shfl_xor_f32(val_f32, offset, width=WARP_SIZE):
            val_i32 = _raw(ArithValue(val_f32).bitcast(T.i32))
            peer_i32 = ArithValue(val_i32).shuffle_xor(offset, width)
            return fx.Float32(ArithValue(peer_i32).bitcast(T.f32))

        def _warp_reduce_max_16(val):
            w = _f32(val)
            for sh in [32, 16]:
                w = _fmax(w, _shfl_xor_f32(w, sh))
            return w

        def _warp_reduce_add_16(val):
            w = _f32(val)
            for sh in [32, 16]:
                w = w + _shfl_xor_f32(w, sh)
            return w

        def _bf16x4dw_to_fp8x2dw(i32x4_bf16):
            f = Vec(Vec(i32x4_bf16).bitcast(fx.BFloat16)).to(fx.Float32)
            fr = [_raw(f[j]) for j in range_constexpr(8)]
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, fr[0], fr[1], c_zero_i32, 0)
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, fr[2], fr[3], w0, 1)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, fr[4], fr[5], c_zero_i32, 0)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, fr[6], fr[7], w1, 1)
            return w0, w1

        # Direct global -> VGPR Q load. The LDS round trip below exists only to
        # transpose from the load layout into the MFMA B layout, but that layout
        # is already contiguous in memory: pack t wants head warp*16 + lane%16
        # and the 8 head-dims at t*32 + (lane/16)*8, i.e. 16 contiguous bytes.
        # Loading it directly puts all 16 loads in flight behind a single wait,
        # instead of 8 serialized rounds each with a ds_write/ds_read pair and
        # three lgkmcnt(0) stalls.
        def _load_q_direct(q_idx_val):
            head = warp_idx * 16 + (lane_idx % 16)
            base_elem = (
                (_idx(q_idx_val) * NUM_QO_HEADS + head) * QK_HEAD_DIM + (lane_idx / 16) * 8
            )
            raws = []
            for t in range_constexpr(NUM_NOPE_ITERS * 2):
                elem = base_elem + t * MFMA_K
                raws.append(
                    buffer_ops.buffer_load(
                        query_rsrc, _i32(ArithValue(elem) // 2), vec_width=4, dtype=T.i32
                    )
                )
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
            packs = []
            for t in range_constexpr(NUM_NOPE_ITERS * 2):
                w0, w1 = _bf16x4dw_to_fp8x2dw(raws[t])
                packs.append(_pack_i32x2(w0, w1))
            return packs


        # ---- Q RoPE bf16 B-operands (split dot) ------------------------------
        # Lane L, step s -> head = warp*16 + L%16, dims = PK_NOPE_DIM + s*16 +
        # (L/16)*4 .. +3 (4 contiguous bf16 -> one i64). No LDS round-trip.
        def _load_q_rope_bf16(q_idx_val):
            head = warp_idx * 16 + (lane_idx % 16)
            base_elem = (
                (_idx(q_idx_val) * NUM_QO_HEADS + head) * QK_HEAD_DIM
                + PK_NOPE_DIM
                + (lane_idx / 16) * 4
            )
            base_dword = _i32(ArithValue(base_elem) // 2)
            pairs = []
            for s in range_constexpr(RBF_NUM_STEPS):
                pairs.append(
                    buffer_ops.buffer_load(
                        query_rsrc,
                        _raw(ArithValue(base_dword) + s * 8),
                        vec_width=2,
                        dtype=T.i32,
                    )
                )
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
            return [_bits_to_i16x4(p) for p in pairs]

        # The 8 CSR positions a lane needs for its P values are two runs of 4
        # consecutive slots (sub-offsets 0..3 and 16..19), so two dwordx4 loads
        # replace eight dword loads on the softmax critical path. Positions past
        # the query's span are masked by ``pos >= kv_end`` exactly as before, so
        # a partially out-of-range vector load is harmless (buffer OOB reads 0).
        SLOT_GROUPS = P_VALS_PER_THR // 4  # 2
        SLOT_GROUP_STRIDE = 16

        def _slot_col0(kv_tile_start_i32):
            return lane_idx / 16 * 4 + _idx(kv_tile_start_i32)

        def _load_slot_groups(idx_rsrc, col_0_start):
            """The 8 CSR slots this lane needs, as a flat i32 list."""
            out = []
            for g in range_constexpr(SLOT_GROUPS):
                v = Vec(
                    buffer_ops.buffer_load(
                        idx_rsrc,
                        _i32(col_0_start + g * SLOT_GROUP_STRIDE),
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for j in range_constexpr(4):
                    out.append(_raw(v[j]))
            return out

        def _softmax_scale_p(num_rows_i32, p_vals, col_0_start, kv_end_i32, slots):
            result = [None] * P_VALS_PER_THR
            for i in range_constexpr(P_VALS_PER_THR):
                result[i] = _f32(p_vals[i]) * qk_softmax_scale
            kv_end = _idx(kv_end_i32)
            skv = ArithValue(num_rows_i32)
            for i in range_constexpr(P_VALS_PER_THR):
                sub_offset = (i // 4) * SLOT_GROUP_STRIDE + (i % 4)
                pos = col_0_start + sub_offset
                slot_a = ArithValue(slots[i])
                inv = ArithValue(_raw(pos >= kv_end))
                inv = inv | (slot_a < 0)
                inv = inv | (slot_a >= skv)
                result[i] = ArithValue(_raw(inv)).select(_raw(c_neg_inf), result[i])
            return result

        def _softmax(
            idx_rsrc, num_rows_i32, p_vals, row_max_old, row_sum_e_old, is_first,
            kv_tile_start_i32, kv_end_i32, slots=None,
        ):
            col_0_start = _slot_col0(kv_tile_start_i32)
            if slots is None:
                slots = _load_slot_groups(idx_rsrc, col_0_start)
            scaled = _softmax_scale_p(num_rows_i32, p_vals, col_0_start, kv_end_i32, slots)
            local_max = scaled[0]
            for i in range_constexpr(1, P_VALS_PER_THR):
                local_max = _fmax(local_max, scaled[i])
            local_max = _warp_reduce_max_16(local_max)
            if const_expr(is_first):
                new_row_max = local_max
                rescale = c_one_f32
            else:
                new_row_max = _fmax(local_max, row_max_old)
                diff = _fsub(row_max_old, new_row_max)
                rescale = _fast_exp2(_fmul(diff, c_log2e))
            p_exp_vals = [None] * P_VALS_PER_THR
            local_sum = c_zero_f32
            for i in range_constexpr(P_VALS_PER_THR):
                exp_arg = _fmul(_fsub(scaled[i], new_row_max), c_log2e)
                p_exp_vals[i] = _fast_exp2(exp_arg)
                local_sum = _fadd(local_sum, p_exp_vals[i])
            local_sum = _warp_reduce_add_16(local_sum)
            if const_expr(is_first):
                row_sum_e_new = local_sum
            else:
                row_sum_e_new = _fadd(_f32(rescale) * row_sum_e_old, local_sum)
            return p_exp_vals, new_row_max, row_sum_e_new, rescale

        def _pack_p_to_fp8(p_exp_vals):
            v = p_exp_vals
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(v[0]), _raw(v[1]), c_zero_i32, 0)
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(v[2]), _raw(v[3]), w0, 1)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(v[4]), _raw(v[5]), c_zero_i32, 0)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(v[6]), _raw(v[7]), w1, 1)
            return _pack_i32x2(w0, w1)

        def _rescale_oaccu(oaccu, rescale):
            rv = _raw(Vec.filled(4, _f32(rescale), fx.Float32))
            return [_f32(oaccu[i]) * rv for i in range_constexpr(len(oaccu))]

        def _process_tile_gemm1(
            idx_rsrc,
            num_rows_i32,
            p_lds_kv_base,
            kv_tile_start_i32,
            kv_end_i32,
            q_nope,
            rm_in,
            rse_in,
            is_first,
            p_lds_kv_next_warp=None,
            prefetch_cache_rsrc=None,
            token_base_next=None,
            q_rope_b=None,
            slots=None,
            rbf_base=None,
        ):
            k_base_i32 = _i32(ArithValue(p_lds_kv_base) + k_lds_lane_offset)
            do_prefetch = p_lds_kv_next_warp is not None

            def _maybe_prefetch(block_idx):
                if const_expr(not do_prefetch):
                    return
                _prefetch_nope_block_asm(
                    prefetch_cache_rsrc, p_lds_kv_next_warp, token_base_next, block_idx
                )

            if const_expr(not KV_PF_LATE):
                _maybe_prefetch(0)
            P_COMP_SUBS = BLOCK_N // MFMA_N
            p_comp = [c_zero_v4f32] * P_COMP_SUBS
            for nope_pair in range_constexpr(NUM_FP8_QK_ITERS):
                tile_0 = nope_pair * 2
                tile_1 = nope_pair * 2 + 1
                k0 = [
                    _load_k_from_lds(k_base_i32, 16 * h, tile_0 * BLOCK_K)
                    for h in range_constexpr(P_COMP_SUBS)
                ]
                k1 = [
                    _load_k_from_lds(k_base_i32, 16 * h, tile_1 * BLOCK_K)
                    for h in range_constexpr(P_COMP_SUBS)
                ]
                if const_expr(nope_pair + 1 < PK_NOPE_BLOCKS and not KV_PF_LATE):
                    _maybe_prefetch(nope_pair + 1)
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=P_COMP_SUBS))
                q_0 = q_nope[tile_0]
                q_1 = q_nope[tile_1]
                if const_expr(nope_pair == 0):
                    for h in range_constexpr(P_COMP_SUBS):
                        p_comp[h] = _mfma_fp8(T.f32x4, [k0[h], q_0, c_zero_v4f32, 0, 0, 0])
                else:
                    for h in range_constexpr(P_COMP_SUBS):
                        p_comp[h] = _mfma_fp8(T.f32x4, [k0[h], q_0, p_comp[h], 0, 0, 0])
                rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                for h in range_constexpr(P_COMP_SUBS):
                    p_comp[h] = _mfma_fp8(T.f32x4, [k1[h], q_1, p_comp[h], 0, 0, 0])
            # bf16 RoPE split dot: accumulate the 64-d rope tail into p_comp[h]
            # using mfma_f32_16x16x16bf16. A = the shared bf16 K tile in
            # P_LDS_RBF; B = q_rope_b.
            if const_expr(ROPE_BF16):
                rbf_lane_base = _i32(
                    ArithValue(_i32(rbf_base))
                    + ArithValue(_i32(lane_idx % 16)) * RBF_ROW_STRIDE
                    + ArithValue(_i32(lane_idx / 16)) * (4 * 2)
                )
                ka = [[None] * RBF_NUM_STEPS for _ in range(P_COMP_SUBS)]
                for h in range_constexpr(P_COMP_SUBS):
                    for s in range_constexpr(RBF_NUM_STEPS):
                        koff = h * MFMA_M * RBF_ROW_STRIDE + s * (RBF_KSTEP * 2)
                        ka[h][s] = _bits_to_i16x4(
                            _lds_load_volatile(rbf_lane_base, T.i32x2, byte_offset=koff)
                        )
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                for h in range_constexpr(P_COMP_SUBS):
                    for s in range_constexpr(RBF_NUM_STEPS):
                        p_comp[h] = _mfma_bf16(p_comp[h], ka[h][s], q_rope_b[s])
            p_vals = []
            for sub in range_constexpr(P_COMP_SUBS):
                pv = Vec(p_comp[sub])
                for ii in range_constexpr(4):
                    p_vals.append(pv[ii])
            # NOTE: issuing this V read before the QK MFMA loop was tried and is
            # ~1.5% SLOWER. lgkmcnt retires LDS ops in order, so hoisting it makes
            # iteration 0's lgkmcnt(P_COMP_SUBS) drain 6 ops instead of 2, at the
            # top of the loop where no MFMA has issued yet to cover it. Here the
            # MFMA pipeline is still draining, which covers more of it.
            if const_expr(VT_WIDE):
                v8_raw = _load_v_rows8(p_lds_kv_base, warp_idx, lane_idx)
                rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            elif const_expr(not VT_INREG):
                v8_raw = _load_v_from_lds(p_lds_kv_base, warp_idx, lane_idx)
                rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            rocdl.sched_barrier(0)
            p_exp_vals, rm_new, rse_new, rescale = _softmax(
                idx_rsrc, num_rows_i32, p_vals, rm_in, rse_in, is_first,
                kv_tile_start_i32, kv_end_i32, slots=slots,
            )
            p_pack = _pack_p_to_fp8(p_exp_vals)
            if const_expr(VT_WIDE):
                _store_vt8_to_lds(
                    lds_base_idx + P_LDS_VT, warp_idx, lane_idx, _transpose_v8(v8_raw)
                )
            elif const_expr(not VT_INREG):
                _store_vt_to_lds(
                    lds_base_idx + P_LDS_VT, warp_idx, lane_idx, _transpose_v(v8_raw)
                )
            if const_expr(KV_PF_LATE):
                for blk in range_constexpr(PK_NOPE_BLOCKS):
                    _maybe_prefetch(blk)
            return rm_new, rse_new, p_pack, rescale

        def _gemm2_core_vt_wide(p_pack, oaccu, vt_base_i32):
            # One ds_read_b64 per operand: the stored bytes are already the
            # complete 8-row A operand.
            for pv_pair in range_constexpr(NUM_PV_ITERS // 2):
                ops = []
                for it in range_constexpr(2):
                    strip = (pv_pair * 2 + it) * MFMA_N * 2
                    ops.append(_load_vt8_from_lds(vt_base_i32, strip))
                    ops.append(_load_vt8_from_lds(vt_base_i32, strip + MFMA_N))
                waits = [2, 0]
                for it in range_constexpr(2):
                    rocdl.sched_barrier(0)
                    rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=waits[it]))
                    acc = (pv_pair * 2 + it) * 2
                    oaccu[acc] = _mfma_fp8(
                        T.f32x4, [ops[it * 2], p_pack, oaccu[acc], 0, 0, 0]
                    )
                    oaccu[acc + 1] = _mfma_fp8(
                        T.f32x4, [ops[it * 2 + 1], p_pack, oaccu[acc + 1], 0, 0, 0]
                    )
                rocdl.sched_barrier(0)
            return oaccu

        def _gemm2_core_vt(p_pack, oaccu, vt_base_i32):
            for pv_pair in range_constexpr(NUM_PV_ITERS // 2):
                iter_a = pv_pair * 2
                iter_b = pv_pair * 2 + 1
                col_a = iter_a * MFMA_N * 2
                col_b = iter_b * MFMA_N * 2
                a0 = _load_vt_from_lds(vt_base_i32, col_a)
                a1 = _load_vt_from_lds(vt_base_i32, col_a + MFMA_N)
                b0 = _load_vt_from_lds(vt_base_i32, col_b)
                b1 = _load_vt_from_lds(vt_base_i32, col_b + MFMA_N)
                lo0 = [a0[0], b0[0]]
                hi0 = [a0[1], b0[1]]
                lo1 = [a1[0], b1[0]]
                hi1 = [a1[1], b1[1]]
                idxs = [iter_a, iter_b]
                waits = [4, 0]
                for step in range_constexpr(2):
                    rocdl.sched_barrier(0)
                    rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=waits[step]))
                    acc = idxs[step] * 2
                    oaccu[acc] = _mfma_fp8(
                        T.f32x4,
                        [_pack_i32x2(lo0[step], hi0[step]), p_pack, oaccu[acc], 0, 0, 0],
                    )
                    oaccu[acc + 1] = _mfma_fp8(
                        T.f32x4,
                        [_pack_i32x2(lo1[step], hi1[step]), p_pack, oaccu[acc + 1], 0, 0, 0],
                    )
                rocdl.sched_barrier(0)
            return oaccu

        def _gemm2_core_inreg(p_pack, oaccu, kv_base):
            # Accumulator s holds v-dim m*32 + s (m = the MFMA's C row), so the
            # 8 operands from column group `grp` feed MFMAs grp*8 .. grp*8+7.
            #
            # Software-pipelined one group deep: group g+1's reads are issued
            # immediately after the wait for group g, so they have g's 32 v_perm
            # plus 8 MFMAs (~256 cycles) to land. Issuing them after the
            # transpose instead leaves the LDS latency fully exposed.
            v_base = _v_inreg_base(kv_base)
            cur = _read_v_raw(v_base, 0)
            for grp in range_constexpr(4):
                rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                nxt = _read_v_raw(v_base, grp + 1) if const_expr(grp + 1 < 4) else None
                rocdl.sched_barrier(0)
                ops = _transpose_v_operands(cur)
                for c in range_constexpr(8):
                    s = grp * 8 + c
                    oaccu[s] = _mfma_fp8(T.f32x4, [ops[c], p_pack, oaccu[s], 0, 0, 0])
                cur = nxt
            if const_expr(not KV_DB):
                # WAR guard: V is read from the KV tile here, so with a single
                # buffer the next tile's load would overwrite it mid-read. The
                # double-buffered path writes the other buffer and does not need
                # this. Omitting it produced a non-deterministic multitile-only
                # mismatch, which is exactly the signature of this race.
                _barrier(lgkmcnt=0)
                rocdl.sched_barrier(0)
            return oaccu


        # There is no V^T to publish, so the barrier that used to
        # guard it before GEMM2 moves to *after* GEMM2. It is still needed
        # because V is now read from the KV tile during GEMM2, and the single KV
        # buffer is overwritten by the next tile's load -- a WAR hazard that
        # barrier B was implicitly covering before. A second KV buffer removes
        # this barrier entirely; until then the count is unchanged.
        def _gemm2_run(p_pack, oaccu, kv_base):
            if const_expr(VT_INREG):
                return _gemm2_core_inreg(p_pack, oaccu, kv_base)
            # Barrier B publishes V^T. It also happens to guard the KV buffer,
            # but KV is double buffered so that is no longer load-bearing.
            _barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)
            if const_expr(VT_WIDE):
                return _gemm2_core_vt_wide(p_pack, oaccu, _vt8_base_i32())
            return _gemm2_core_vt(p_pack, oaccu, _vt_base_i32())

        def _gemm2_first_iter(p_pack, kv_base):
            return _gemm2_run(p_pack, [c_zero_v4f32] * (NUM_PV_ITERS * 2), kv_base)

        def _gemm2_with_rescale(p_pack, rescale, oaccu_in, kv_base):
            return _gemm2_run(p_pack, _rescale_oaccu(oaccu_in, rescale), kv_base)

        def _pack_f32x4_to_bf16_2dw(acc_val):
            i16s = Vec(acc_val).to(fx.BFloat16).bitcast(fx.Int16)
            i16_0, i16_1, i16_2, i16_3 = (_raw(i16s[j]) for j in range(4))
            dw0 = _raw(ArithValue(i16_0).extui(T.i32) | (ArithValue(i16_1).extui(T.i32) << 16))
            dw1 = _raw(ArithValue(i16_2).extui(T.i32) | (ArithValue(i16_3).extui(T.i32) << 16))
            return dw0, dw1

        # Epilogue pass: each lane contributes 8 contiguous v-dims at
        # column (lane/16)*8 of a 16 head x 32 v-dim tile, which the LDS reshape
        # turns into the same coalesced 16-byte store the default path makes.
        def _o16_addrs(p_lds_o):
            row_st = lane_idx % 16
            col_st = (lane_idx / 16) * 4
            row_ld = lane_idx / 4
            col_ld = (lane_idx % 4) * 8
            warp_base = ArithValue(p_lds_o) + warp_idx * O16_LDS_PER_WARP
            st = _raw(
                ((row_st / 2) * O16_ELEM_PER_PAD_2ROWS + (row_st % 2) * O16_NUM_COLS + col_st)
                * 2
            )
            rd = _raw(
                ((row_ld / 2) * O16_ELEM_PER_PAD_2ROWS + (row_ld % 2) * O16_NUM_COLS + col_ld)
                * 2
            )
            return warp_base, st, rd, row_ld, col_ld

        # Default (V^T) epilogue pass: two accumulators, 4 contiguous v-dims each.
        def _store_o4x2_bf16(dws, col_base, p_lds_o, row_base_i32):
            warp_base, st, rd, row_ld, col_ld = _o16_addrs(p_lds_o)
            st_addr = _i32(ArithValue(warp_base) + st)
            for sub_i in range_constexpr(2):
                _ptr_store(
                    Vec.from_elements(dws[sub_i * 2 : sub_i * 2 + 2], fx.Int32),
                    _lds_ptr_from_i32(_i32(ArithValue(st_addr) + sub_i * O16_NUM_COLS)),
                    alignment=8,
                    volatile_=True,
                )
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            data = _ptr_load(
                T.i32x4, _lds_ptr_from_i32(_i32(ArithValue(warp_base) + rd)), alignment=16
            )
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            row_vram = ArithValue(row_base_i32) + row_ld
            col_vram = ArithValue(col_ld) + col_base
            buffer_ops.buffer_store(
                data,
                final_output_rsrc,
                _raw((row_vram * V_HEAD_DIM + col_vram) * 2),
                offset_is_bytes=True,
            )

        def _store_o8_bf16(dws4, col_base, p_lds_o, row_base_i32):
            row_st = lane_idx % 16
            col_st = (lane_idx / 16) * 8
            st_offset = _raw(
                ((row_st / 2) * O16_ELEM_PER_PAD_2ROWS + (row_st % 2) * O16_NUM_COLS + col_st)
                * 2
            )
            row_ld = lane_idx / 4
            col_ld = (lane_idx % 4) * 8
            rd_offset = _raw(
                ((row_ld / 2) * O16_ELEM_PER_PAD_2ROWS + (row_ld % 2) * O16_NUM_COLS + col_ld)
                * 2
            )
            lds_warp = ArithValue(p_lds_o) + warp_idx * O16_LDS_PER_WARP
            _ptr_store(
                Vec.from_elements(dws4, fx.Int32),
                _lds_ptr_from_i32(_i32(ArithValue(lds_warp) + st_offset)),
                alignment=16,
                volatile_=True,
            )
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            data = _ptr_load(
                T.i32x4,
                _lds_ptr_from_i32(_i32(ArithValue(lds_warp) + rd_offset)),
                alignment=16,
            )
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            row_vram = ArithValue(row_base_i32) + row_ld
            col_vram = ArithValue(col_ld) + col_base
            vram_offset = _raw((row_vram * V_HEAD_DIM + col_vram) * 2)
            buffer_ops.buffer_store(data, final_output_rsrc, vram_offset, offset_is_bytes=True)


        def _normalize_and_store(oaccu, rm, rse, row_base_idx):
            p_lds_o = p_lds_kv_0_base
            # sink: fold a per-head virtual key (score=sink[h], zero value).
            if const_expr(HAS_SINK):
                head = _i32(
                    ArithValue(_uniform_i32(warp_idx)) * 16 + ArithValue(_i32(lane_idx % 16))
                )
                sink_val = _f32(
                    buffer_ops.buffer_load(sink_rsrc, head, vec_width=1, dtype=T.f32)
                )
                m_fin = _fmax(rm, sink_val)
                alpha = _fast_exp2(_fmul(_fsub(rm, m_fin), c_log2e))
                sink_term = _fast_exp2(_fmul(_fsub(sink_val, m_fin), c_log2e))
                l_fin = _fadd(_fmul(_f32(rse), alpha), sink_term)
            else:
                alpha = c_one_f32
                l_fin = _f32(rse)
            valid_i1 = _raw(ArithValue(_raw(l_fin)) > c_zero_f32)
            denom = _fmax(l_fin, fx.Float32(1e-30))
            reci = rocdl.rcp(T.f32, _raw(denom))
            scl = _fmul(alpha, reci) if const_expr(HAS_SINK) else _f32(reci)
            scl_vec = _raw(Vec.filled(4, _f32(scl), fx.Float32))
            if const_expr(not VT_INREG):
                # Default layout: accumulator a holds v-dims [16a, 16a+16).
                _barrier(lgkmcnt=0)
                for pv in range_constexpr(NUM_PV_ITERS):
                    pair = []
                    for sub_i in range_constexpr(2):
                        v = _f32(oaccu[pv * 2 + sub_i]) * scl_vec
                        pair.append(
                            ArithValue(_raw(valid_i1)).select(_raw(v), _raw(c_zero_v4f32))
                        )
                    dws = []
                    for sub_i in range_constexpr(2):
                        d0, d1 = _pack_f32x4_to_bf16_2dw(pair[sub_i])
                        dws.append(d0)
                        dws.append(d1)
                    _store_o4x2_bf16(dws, pv * MFMA_N * 2, p_lds_o, row_base_idx)
                return
            # Accumulator 8*grp + c holds v-dim 128*grp + 32*j + 8*(lane/16) + c,
            # so each (grp, j) pass covers one contiguous 16 head x 32 v-dim
            # tile -- the shape the OManager reshape stages, which keeps the
            # output stores coalesced.
            _barrier(lgkmcnt=0)
            for grp in range_constexpr(4):
                for j in range_constexpr(4):
                    vals = []
                    for c in range_constexpr(8):
                        vals.append(_raw(Vec(oaccu[grp * 8 + c])[j]))
                    halves = []
                    for h in range_constexpr(2):
                        quad = Vec.from_elements(vals[h * 4 : h * 4 + 4], fx.Float32)
                        scaled = _f32(_raw(quad)) * scl_vec
                        scaled = ArithValue(_raw(valid_i1)).select(
                            _raw(scaled), _raw(c_zero_v4f32)
                        )
                        halves.append(_pack_f32x4_to_bf16_2dw(_raw(scaled)))
                    dws = [halves[0][0], halves[0][1], halves[1][0], halves[1][1]]
                    _store_o8_bf16(dws, grp * 128 + j * 32, p_lds_o, row_base_idx)

        p_lds_kv_0_base = lds_base_idx + P_LDS_KV_0
        p_lds_kv_1_base = lds_base_idx + P_LDS_KV_1

        def _kv_warp_lds_base(p_lds_kv_base):
            warp_offset = _raw(ArithValue(_uniform_i32(warp_idx)) * KV_SUB_BYTES)
            return _raw(ArithValue(_i32(p_lds_kv_base)) + warp_offset)

        p_lds_kv_0_warp = _kv_warp_lds_base(p_lds_kv_0_base)
        p_lds_kv_1_warp = _kv_warp_lds_base(p_lds_kv_1_base)

        def p_lds_rbf_base(slot):
            # Only read under ROPE_BF16; the region is not allocated otherwise.
            return _i32(lds_base_idx + P_LDS_RBF + slot * SZ_LDS_RBF)

        # ---- CSR ranges ----
        main_rng = Vec(
            buffer_ops.buffer_load(main_indptr_rsrc, q_idx, vec_width=2, dtype=T.i32)
        )
        main_start = rocdl.readfirstlane(T.i32, main_rng[0])
        main_end = rocdl.readfirstlane(T.i32, main_rng[1])
        main_len = _raw(ArithValue(main_end) - ArithValue(main_start))
        n0_tiles = _raw((ArithValue(main_len) + (BLOCK_N - 1)).with_signedness(False) // BLOCK_N)

        extra_rng = Vec(
            buffer_ops.buffer_load(extra_indptr_rsrc, q_idx, vec_width=2, dtype=T.i32)
        )
        extra_start = rocdl.readfirstlane(T.i32, extra_rng[0])
        extra_end = rocdl.readfirstlane(T.i32, extra_rng[1])
        extra_len = _raw(ArithValue(extra_end) - ArithValue(extra_start))
        n1_tiles = _raw((ArithValue(extra_len) + (BLOCK_N - 1)).with_signedness(False) // BLOCK_N)
        total_tiles = _raw(ArithValue(n0_tiles) + ArithValue(n1_tiles))

        if const_expr(R1_TB_CARRY):
            # The r0->r1 boundary tile cannot take the carry: the trailing region0
            # tile computes tb_next against main_* and gets -1 past main_end. But
            # that tile's own base is a per-query constant -- for local==0,
            # kv_ts == extra_start and _row_addrs then depends only on prologue
            # values -- so resolve it once here instead. The region1 arm can then
            # pick between this and the carry with a v_cndmask rather than the
            # dynamic branch that made the first attempt at this 1.37% slower.
            tb_r1_head = _row_addrs(
                extra_indices_rsrc, extra_bt_rsrc, extra_num_rows, extra_block_size,
                extra_max_blocks, extra_start, extra_end,
            )[0]

        row_base = _idx(q_idx) * NUM_QO_HEADS + warp_idx * 16
        # Tile 0's CSR slot goes out before the Q loads so the two round trips
        # overlap and ``_load_q_direct``'s vmcnt(0) covers both. Left in series
        # they cost the Q drain plus the slot drain back to back; this is the
        # coldest chain in the kernel (nothing else is in flight yet), which is
        # why it is the most expensive _row_addrs instance despite running once.
        if const_expr(SLOT_HOIST):
            slot0_pre = _issue_row_slot(main_indices_rsrc, main_start)
        else:
            slot0_pre = None
        q_nope_packs = _load_q_direct(q_idx)
        q_rope_packs = _load_q_rope_bf16(q_idx) if const_expr(ROPE_BF16) else None

        # NOTE: a one-tile CSR slot lookahead (issue tile g+1's slot loads during
        # tile g, carry them in the loop state) was implemented and measured as a
        # net ~1% LOSS, so it is deliberately not here. ATT confirmed it did what
        # it was meant to -- the 730K-cycle softmax slot wait vanished and total
        # stall fell 5.4% -- but wall time got worse because at 2 waves/SIMD the
        # co-resident wave was already covering that stall, while the freed waves
        # then piled up at the CTA barrier (barrier A doubled, 331K -> 689K).
        # Per-wave latency hiding is not the lever here; barrier count is.

        # ---- region-0 attend body (const-fixed resources) ----
        def _attend_region0(kv_tile_start_i32, rm_in, rse_in, oaccu_in, is_first,
                           slot_pre=None):
            tb, sb = _row_addrs(
                main_indices_rsrc, main_bt_rsrc, main_num_rows, main_block_size,
                main_max_blocks, kv_tile_start_i32, main_end, slot_pre=slot_pre,
            )
            if const_expr(R0_CONVERT):
                _load_nope_convert(
                    main_cache_rsrc, p_lds_kv_0_warp, tb, sb,
                    fx.Float32(1.0) if R0_OCP else fx.Float32(0.0),
                )
            else:
                _load_nope_dma(main_cache_rsrc, p_lds_kv_0_warp, tb)
            if const_expr(ROPE_FP8):
                _dma_rope_block(main_cache_rsrc, p_lds_kv_0_warp, tb)
            else:
                _load_rope_block(main_cache_rsrc, p_lds_kv_0_warp, tb, p_lds_rbf_base(0))
            # Tile 1's NoPE blocks are prefetched into buffer 1 during this
            # tile's QK MFMAs; region0 always owns tile 0, so the resource is
            # compile-time fixed. Past the end of region0 the token base is -1
            # and the prefetch degenerates to writing zeros, which the next tile
            # overwrites (it loads synchronously if it is a region1 tile).
            kv_ts_next = _raw(ArithValue(kv_tile_start_i32) + BLOCK_N)
            tb_next, _sb_next = _row_addrs(
                main_indices_rsrc, main_bt_rsrc, main_num_rows, main_block_size,
                main_max_blocks, kv_ts_next, main_end,
            )
            _barrier(vmcnt=0, lgkmcnt=0)
            rocdl.sched_barrier(0)
            pf_kwargs = {}
            if const_expr(KV_DB and not R0_CONVERT):
                pf_kwargs = dict(
                    p_lds_kv_next_warp=p_lds_kv_1_warp,
                    prefetch_cache_rsrc=main_cache_rsrc,
                    token_base_next=tb_next,
                )
            rm_n, rse_n, p_pack, rescale = _process_tile_gemm1(
                main_indices_rsrc, main_num_rows, p_lds_kv_0_base, kv_tile_start_i32, main_end,
                q_nope_packs, rm_in, rse_in, is_first, q_rope_b=q_rope_packs,
                rbf_base=p_lds_rbf_base(0), **pf_kwargs,
            )
            # Issue tile 1's RoPE load after the softmax, so the softmax's own
            # slot loads do not drag it in via the in-order vmcnt, and nothing
            # before the next tile's barrier waits on it.
            if const_expr(ROPE_PF and not ROPE_FP8):
                nxt = Vec(_issue_rope_load(main_cache_rsrc, tb_next))
                rp0_n = nxt[0]
                rp1_n = nxt[1]
            else:
                rp0_n = c_zero_i32
                rp1_n = c_zero_i32
            tb_n = tb_next if const_expr(TB_CARRY) else c_zero_i32
            if const_expr(is_first):
                oaccu_n = _gemm2_first_iter(p_pack, p_lds_kv_0_base)
            else:
                oaccu_n = _gemm2_with_rescale(p_pack, rescale, oaccu_in, p_lds_kv_0_base)
            return rm_n, rse_n, oaccu_n, rp0_n, rp1_n, tb_n

        # ---- region-select load + GEMM1 ----
        # arith.select on the !llvm.ptr<8> buffer descriptors does NOT lower
        # correctly, and two sequential yield-loops in one body break the
        # structured-for lowering. So this uses ONE yield-loop with a runtime
        # ``if`` that selects the region for the load + GEMM1 (each region keeps
        # compile-time-fixed resources). Only flat scalars cross the if
        # (rm, rse, the i64 P-pack, rescale) -- the GEMM2 P@V accumulation reads
        # V^T from LDS and is region-agnostic, so it runs after the if.
        # Tile g lives in KV buffer g&1. Region0 tiles have their NoPE blocks
        # DMA'd in during tile g-1's QK MFMAs; the RoPE tail (register-staged, so
        # not a fire-and-forget DMA) and all of region1 (which needs the convert
        # path) still load synchronously at the top of the tile. One barrier per
        # tile covers both hazards: RAW on buf[g&1], and WAR on buf[(g+1)&1]
        # whose last reader was tile g-1's GEMM2.
        def _load_gemm1_select(global_t_i32, rm_in, rse_in, is_first, rp0, rp1, tb_c):
            is_r1 = _raw(ArithValue(global_t_i32) >= ArithValue(n0_tiles))
            # Without double buffering everything stays in buffer 0, which keeps
            # the K/V base addresses loop-invariant so they hoist out of the tile
            # loop. Alternating makes them a runtime select recomputed per tile.
            if const_expr(KV_DB):
                is_odd = (ArithValue(global_t_i32) & 1) != 0
                cur_base = ArithValue(is_odd).select(p_lds_kv_1_base, p_lds_kv_0_base)
                cur_warp = ArithValue(is_odd).select(p_lds_kv_1_warp, p_lds_kv_0_warp)
                next_warp = ArithValue(is_odd).select(p_lds_kv_0_warp, p_lds_kv_1_warp)
                cur_rbf = ArithValue(is_odd).select(p_lds_rbf_base(1), p_lds_rbf_base(0))
            else:
                cur_base = p_lds_kv_0_base
                cur_warp = p_lds_kv_0_warp
                next_warp = p_lds_kv_1_warp
                cur_rbf = p_lds_rbf_base(0)
            rm_n = c_neg_large
            rse_n = c_zero_f32
            pp = fx.Int64(0)
            rescale = c_one_f32
            # The RoPE pair is region0-only (region1 either DMAs the rope or loads it
            # synchronously), so it passes through untouched on this side.
            nrp0 = rp0
            nrp1 = rp1
            ntb = tb_c
            if is_r1:
                local = _raw(ArithValue(global_t_i32) - ArithValue(n0_tiles))
                kv_ts = _raw(ArithValue(extra_start) + ArithValue(local) * BLOCK_N)
                sb = c_zero_i32
                tb = c_zero_i32
                if const_expr(R1_TB_CARRY):
                    # Resolving the address inline costs region1 615 cyc/tile at the
                    # tile barrier against region0's 169: the DMA cannot issue until
                    # the chain lands, so the barrier waits on it. Carrying the base
                    # lets the DMA go out at the top of the tile as region0's does.
                    # The boundary tile takes tb_r1_head, hoisted to the prologue, so
                    # this stays a select instead of splitting the arm in two.
                    tb = _raw(
                        ArithValue(_raw(ArithValue(local) == 0)).select(
                            _raw(tb_r1_head), _raw(tb_c)
                        )
                    )
                else:
                    tb, sb = _row_addrs(
                        extra_indices_rsrc, extra_bt_rsrc, extra_num_rows,
                        extra_block_size, extra_max_blocks, kv_ts, extra_end,
                    )
                if const_expr(R1_CONVERT):
                    _load_nope_convert(
                        extra_cache_rsrc, cur_warp, tb, sb,
                        fx.Float32(1.0) if R1_OCP else fx.Float32(0.0),
                    )
                else:
                    _load_nope_dma(extra_cache_rsrc, cur_warp, tb)
                if const_expr(ROPE_FP8):
                    _dma_rope_block(extra_cache_rsrc, cur_warp, tb)
                else:
                    _load_rope_block(extra_cache_rsrc, cur_warp, tb, cur_rbf)
                if const_expr(R1_TB_CARRY):
                    kv_ts_next = _raw(ArithValue(kv_ts) + BLOCK_N)
                    ntb = _row_addrs(
                        extra_indices_rsrc, extra_bt_rsrc, extra_num_rows,
                        extra_block_size, extra_max_blocks, kv_ts_next, extra_end,
                    )[0]
                _barrier(vmcnt=0, lgkmcnt=0)
                rocdl.sched_barrier(0)
                rm_n, rse_n, pp, rescale = _process_tile_gemm1(
                    extra_indices_rsrc, extra_num_rows, cur_base, kv_ts, extra_end,
                    q_nope_packs, rm_in, rse_in, is_first, q_rope_b=q_rope_packs,
                    rbf_base=cur_rbf,
                )
            else:
                kv_ts = _raw(ArithValue(main_start) + ArithValue(global_t_i32) * BLOCK_N)
                if const_expr(TB_CARRY):
                    # Computed as tb_next during the previous tile.
                    tb = tb_c
                    sb = c_zero_i32
                else:
                    tb, sb = _row_addrs(
                        main_indices_rsrc, main_bt_rsrc, main_num_rows, main_block_size,
                        main_max_blocks, kv_ts, main_end,
                    )
                if const_expr(R0_CONVERT):
                    _load_nope_convert(
                        main_cache_rsrc, cur_warp, tb, sb,
                        fx.Float32(1.0) if R0_OCP else fx.Float32(0.0),
                    )
                elif const_expr(not KV_DB):
                    _load_nope_dma(main_cache_rsrc, cur_warp, tb)
                if const_expr(ROPE_FP8):
                    _dma_rope_block(main_cache_rsrc, cur_warp, tb)
                elif const_expr(ROPE_PF):
                    _commit_rope_block(
                        cur_warp, tb, cur_rbf,
                        _raw(Vec.from_elements([rp0, rp1], fx.Int32)),
                    )
                else:
                    _load_rope_block(main_cache_rsrc, cur_warp, tb, cur_rbf)
                kv_ts_next = _raw(ArithValue(kv_ts) + BLOCK_N)
                tb_next, _sb_next = _row_addrs(
                    main_indices_rsrc, main_bt_rsrc, main_num_rows, main_block_size,
                    main_max_blocks, kv_ts_next, main_end,
                )
                _barrier(vmcnt=0, lgkmcnt=0)
                rocdl.sched_barrier(0)
                pf_kwargs = {}
                if const_expr(KV_DB and not R0_CONVERT):
                    pf_kwargs = dict(
                        p_lds_kv_next_warp=next_warp,
                        prefetch_cache_rsrc=main_cache_rsrc,
                        token_base_next=tb_next,
                    )
                rm_n, rse_n, pp, rescale = _process_tile_gemm1(
                    main_indices_rsrc, main_num_rows, cur_base, kv_ts, main_end,
                    q_nope_packs, rm_in, rse_in, is_first, q_rope_b=q_rope_packs,
                    rbf_base=cur_rbf, **pf_kwargs,
                )
                if const_expr(ROPE_PF and not ROPE_FP8):
                    nxt = Vec(_issue_rope_load(main_cache_rsrc, tb_next))
                    nrp0 = nxt[0]
                    nrp1 = nxt[1]
                if const_expr(TB_CARRY):
                    ntb = tb_next
            return rm_n, rse_n, pp, rescale, cur_base, nrp0, nrp1, ntb

        # Single yield-loop over the flattened region0||region1 tile space. The
        # first tile (g==0) is region0 tile 0 (region0 is the always-present
        # pool) and initialises the shared softmax state via GEMM2-first.
        rm_first, rse_first, oaccu_first, rp0_first, rp1_first, tb_first = _attend_region0(
            main_start, c_neg_large, c_zero_f32, None, True, slot_pre=slot0_pre
        )
        has_multi = ArithValue(total_tiles) > 1
        N_ACC = NUM_PV_ITERS * 2

        def _multi_tile_path():
            init_args = (
                [rm_first, rse_first] + oaccu_first + [rp0_first, rp1_first, tb_first]
            )
            for tile_iv, state in range(_idx(1), _idx(total_tiles), _idx(1), init=init_args):
                tile_i32 = _raw(ArithValue(fx.Int32(tile_iv)))
                rm_c = state[0]
                rse_c = state[1]
                oaccu_c = [state[2 + i] for i in range(N_ACC)]
                rp0_c = state[2 + N_ACC]
                rp1_c = state[3 + N_ACC]
                tb_cc = state[4 + N_ACC]
                (
                    rm_n, rse_n, pp, rescale, cur_base, rp0_n, rp1_n, tb_n
                ) = _load_gemm1_select(
                    tile_i32, rm_c, rse_c, False, rp0_c, rp1_c, tb_cc
                )
                oaccu_n = _gemm2_with_rescale(pp, rescale, oaccu_c, cur_base)
                results = yield [rm_n, rse_n] + oaccu_n + [rp0_n, rp1_n, tb_n]
            rm_final = results[0]
            rse_final = results[1]
            oaccu_final = [results[2 + i] for i in range(N_ACC)]
            _normalize_and_store(oaccu_final, rm_final, rse_final, row_base)

        def _single_tile_path():
            _normalize_and_store(oaccu_first, rm_first, rse_first, row_base)

        @flyc.jit
        def _dispatch():
            if has_multi:
                _multi_tile_path()
            else:
                _single_tile_path()

        _dispatch()

    @flyc.jit
    def launch_dsv4(
        query: fx.Tensor,
        main_cache: fx.Tensor,
        main_indices: fx.Tensor,
        main_indptr: fx.Tensor,
        main_block_table: fx.Tensor,
        extra_cache: fx.Tensor,
        extra_indices: fx.Tensor,
        extra_indptr: fx.Tensor,
        extra_block_table: fx.Tensor,
        q_req: fx.Tensor,
        sink_buf: fx.Tensor,
        final_output: fx.Tensor,
        num_queries: fx.Int32,
        main_num_rows: fx.Int32,
        extra_num_rows: fx.Int32,
        main_block_size: fx.Int32,
        extra_block_size: fx.Int32,
        main_max_blocks: fx.Int32,
        extra_max_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = arith.index_cast(T.index, _raw(num_queries))
        kn_sparse_mla_prefill_dsv4(
            query, main_cache, main_indices, main_indptr, main_block_table,
            extra_cache, extra_indices, extra_indptr, extra_block_table,
            q_req, sink_buf, final_output, SOFTMAX_SCALE,
            main_num_rows, extra_num_rows, main_block_size, extra_block_size,
            main_max_blocks, extra_max_blocks,
        ).launch(grid=(grid_x, 1, 1), block=(NUM_THREADS, 1, 1), smem=0, stream=stream)

    return launch_dsv4
