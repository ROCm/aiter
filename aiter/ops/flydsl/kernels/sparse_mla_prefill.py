# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL sparse MLA prefill kernel (gfx942 / CDNA3) -- Phase A MVP.

Single-region, native fp8 MFMA (``mfma_f32_16x16x32_fp8_fp8``) attention over a
CSR-gathered subset of MLA-latent KV rows. One CTA per query token; 8 warps
(512 threads) process all 128 Q heads of that token, matching the MLA-V2 decode
template (``mla_fwd_decode_m16x8_fp8_fp8.py``) from which the LDS layout,
software V-transpose, fp8 MFMA chains, and online softmax are ported.

Phase A scope (see docs/sparse-mla-prefill/01,02):
  - HEAD_DIM = 512, V_DIM = 512, NUM_REGIONS = 1, HAS_SINK = False.
  - Flat HEAD_DIM dot (no nope/rope qk_split).
  - Native fp8 MFMA for both GEMMs. Q is provided bf16 and cast to fp8 in the
    loader; KV is provided as e4m3fnuz fp8 bytes (uint8). The UE8M0 per-block
    scale fold is a no-op in Phase A (scale == 1, kv already quantized); the
    packed fp8_ds_mla cache + per-64-block scale region is Phase B.
  - CSR gather: flat int32 ``indices`` + ``indptr[num_queries+1]``. Invalid
    slots (slot < 0 or slot >= num_kv_rows) and positions >= kv_end score
    -inf and contribute nothing. Empty rows (kv_len == 0) emit a zero row.
  - BLOCK_N = 32, single KV LDS buffer (no software prefetch pipeline),
    software V-transpose via ``v_perm_b32``.
  - No split-KV.

gfx942 only -- no gfx950 intrinsics (no ds_read_b64_tr, no wide DMA).

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
# Compile-time constants (Phase A: HEAD_DIM=512, V_DIM=512, no rope, no sink)
# ---------------------------------------------------------------------------
NUM_QO_HEADS: int = 128
NUM_KV_HEADS: int = 1
KV_LORA_RANK: int = 512
QK_HEAD_DIM: int = 512  # flat dot over the whole latent
V_HEAD_DIM: int = 512
NUM_WARPS: int = 8
WARP_SIZE: int = 64
NUM_THREADS: int = NUM_WARPS * WARP_SIZE  # 512
BLOCK_M: int = 128  # == NUM_QO_HEADS
BLOCK_N: int = 32
BLOCK_K: int = 32
TILE_M: int = BLOCK_M // NUM_WARPS  # 16
LOG2E: float = 1.4426950408889634

# ---- KvManagerV2 LDS layout (32 rows x 512 cols fp8, 8 blocks of 64 cols) ----
KV_NUM_COLS: int = 64
KV_NUM_BLOCKS: int = QK_HEAD_DIM // KV_NUM_COLS  # 8
KV_ROWS_PER_SUB: int = BLOCK_N // NUM_WARPS  # 4
KV_BYTES_PER_ROW: int = KV_NUM_COLS  # 64 (fp8)
KV_PAD_DW: int = 2
KV_SUB_BYTES: int = KV_ROWS_PER_SUB * KV_BYTES_PER_ROW + KV_PAD_DW * 4  # 264
KV_NUM_SUBS: int = BLOCK_N // KV_ROWS_PER_SUB  # 8
KV_BLOCK_BYTES: int = KV_SUB_BYTES * KV_NUM_SUBS  # 2112
SZ_LDS_KV: int = KV_BLOCK_BYTES * KV_NUM_BLOCKS  # 2112 * 8 = 16896

# ---- VtManagerV1 LDS layout (software-transposed V staging) ----
VT_ROWS_PER_THR: int = 4
VT_COLS_PER_THR: int = 8
VT_ELEMS_PER_BLK: int = VT_ROWS_PER_THR * VT_COLS_PER_THR  # 32
VT_BLKS_PER_ROW: int = V_HEAD_DIM // VT_COLS_PER_THR  # 64
VT_BLKS_PER_ROW_PAD: int = VT_BLKS_PER_ROW + 2  # 66
VT_NUM_SUB_BLKS: int = 8
SZ_LDS_VT: int = VT_NUM_SUB_BLKS * ((BLOCK_N // VT_NUM_SUB_BLKS) * V_HEAD_DIM + 16 * 4)  # 16896

# ---- QManagerV3 LDS layout (per-warp staging for VRAM->LDS->GPR) ----
Q_ELEM_PER_ROW: int = 64
Q_ELEM_PER_COL: int = 16
Q_PAD_BYTES_PER_2ROWS: int = 8
Q_BYTES_PER_2ROWS: int = Q_ELEM_PER_ROW * 2 + Q_PAD_BYTES_PER_2ROWS  # 136
SZ_LDS_Q_PER_WARP: int = Q_ELEM_PER_COL // 2 * Q_BYTES_PER_2ROWS  # 1088
SZ_LDS_Q: int = NUM_WARPS * SZ_LDS_Q_PER_WARP  # 8704
Q_NUM_PASSES: int = QK_HEAD_DIM // Q_ELEM_PER_ROW  # 8

# ---- OManager16bitsV2 (bf16 output via LDS reshape; reuses KV buffer 0) ----
O16_NUM_ROWS: int = 16
O16_NUM_COLS: int = 32
O16_PAD_ELEM_PER_2ROWS: int = 4
O16_ELEM_PER_PAD_2ROWS: int = 2 * O16_NUM_COLS + O16_PAD_ELEM_PER_2ROWS  # 68
O16_LDS_PER_WARP: int = (O16_NUM_ROWS // 2) * O16_ELEM_PER_PAD_2ROWS * 2  # 1088
SZ_LDS_O16: int = NUM_WARPS * O16_LDS_PER_WARP  # 8704

# ---- Overall LDS layout (byte offsets) ----
P_LDS_VT: int = 0
P_LDS_Q: int = SZ_LDS_VT  # 16896
P_LDS_KV_0: int = P_LDS_Q + SZ_LDS_Q  # 25600
TOTAL_LDS_BYTES: int = P_LDS_KV_0 + SZ_LDS_KV  # 42496

assert TOTAL_LDS_BYTES <= 65536, f"gfx942 LDS cap: need {TOTAL_LDS_BYTES} B"
assert SZ_LDS_O16 <= SZ_LDS_KV, "Output LDS must fit in the KV buffer region"

# ---- MFMA tile constants ----
MFMA_M: int = 16
MFMA_N: int = 16
MFMA_K: int = 32  # mfma_f32_16x16x32_fp8_fp8
MFMA_ELEM_PER_THR: int = MFMA_M * MFMA_K // WARP_SIZE  # 8

NUM_NOPE_ITERS: int = QK_HEAD_DIM // (MFMA_K * 2)  # 512/64 = 8
NUM_PV_ITERS: int = V_HEAD_DIM // (MFMA_N * 2)  # 512/32 = 16
P_VALS_PER_THR: int = (BLOCK_N * MFMA_M) // WARP_SIZE  # 8

# ---------------------------------------------------------------------------
# Phase B: fp8_ds_mla packed-cache layout (docs/sparse-mla-prefill/01 Sec 6.1)
# ---------------------------------------------------------------------------
# Packed uint8 cache: [num_blocks, block_size, 584].
#   token_data @ block_base + pos*576 : 448 fp8 NoPE + 128 B (64 bf16) RoPE
#   scales     @ block_base + block_size*576 + pos*8 : 7 UE8M0 bytes (+1 pad)
PK_NOPE_DIM: int = 448
PK_ROPE_DIM: int = 64
PK_TOKEN_BYTES: int = 576  # 448 fp8 + 128 bf16
PK_NOPE_BYTES: int = 448
PK_CACHE_ROW: int = 584  # per-token block stride contribution (576 data + 8 scale)
PK_NOPE_BLOCKS: int = PK_NOPE_DIM // KV_NUM_COLS  # 7 (cols 0..447)
PK_ROPE_BLOCK: int = PK_NOPE_BLOCKS  # block index 7 holds the requant'd RoPE tail
NEG_LARGE: float = -3.4028234663852886e38


# ---------------------------------------------------------------------------
# Module-level utility helpers (ported verbatim from the MLA decode template)
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
    # compiles in a new context and reusing the cached Type fails verification
    # ("result type from a different MLIRContext").
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
# Kernel
# ---------------------------------------------------------------------------
@flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
def kn_sparse_mla_prefill(
    # --- inputs ---
    query: fx.Tensor,  # [num_queries * num_heads, head_dim]  (bf16)
    kv_buffer: fx.Tensor,  # [num_kv_rows, head_dim]  (fp8 e4m3fnuz, as uint8)
    indices: fx.Tensor,  # [nnz]  (i32, CSR column ids)
    indptr: fx.Tensor,  # [num_queries + 1]  (i32, CSR offsets)
    # --- outputs ---
    final_output: fx.Tensor,  # [num_queries * num_heads, v_dim]  (bf16)
    # --- parameters ---
    softmax_scale: fx.Float32,
    num_kv_rows: fx.Int32,
):
    """Sparse MLA prefill: one CTA per query token, 8 warps over 128 heads."""
    # ---- fastmath flags ----
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

    def _fadd(a, b):
        return arith.addf(_raw(a), _raw(b), fastmath=fm_no_inf)

    def _fsub(a, b):
        return arith.subf(_raw(a), _raw(b), fastmath=fm_no_inf)

    def _fmul(a, b):
        return arith.mulf(_raw(a), _raw(b), fastmath=fm_no_inf)

    def _fmax(a, b):
        return arith.maximumf(_raw(a), _raw(b), fastmath=fm_no_inf)

    # ---- LDS setup ----
    arch = get_hip_arch()
    lds_allocator = SmemAllocator(None, arch=arch)
    lds_allocator.ptr = TOTAL_LDS_BYTES
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        lds_allocator.finalize()
    lds_buffer = lds_allocator.get_base()
    lds_base_idx = memref.extract_aligned_pointer_as_index(lds_buffer)

    # ---- V^T transpose perm constants ----
    c_perm0 = fx.Int32(0x05010400)
    c_perm1 = fx.Int32(0x07030602)
    c_perm2 = fx.Int32(0x05040100)
    c_perm3 = fx.Int32(0x07060302)

    def _vt_perm(src_hi, src_lo, sel):
        return rocdl.perm_b32(src_hi, src_lo, sel)

    # ---- Constants ----
    c_neg_inf = fx.Float32(float("-inf"))
    c_zero_f32 = fx.Float32(0.0)
    c_one_f32 = fx.Float32(1.0)
    c_zero_i32 = fx.Int32(0)
    c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)
    c_zero_v4i32 = Vec.filled(4, 0, fx.Int32)
    c_log2e = fx.Float32(LOG2E)

    # ---- Buffer resources ----
    query_rsrc = buffer_ops.create_buffer_resource(query)
    kv_rsrc = buffer_ops.create_buffer_resource(kv_buffer)
    indices_rsrc = buffer_ops.create_buffer_resource(indices)
    indptr_rsrc = buffer_ops.create_buffer_resource(indptr)
    final_output_rsrc = buffer_ops.create_buffer_resource(final_output)

    # ---- Thread indices ----
    q_idx = gpu.block_idx.x
    tid = gpu.thread_id("x")
    warp_idx = tid / WARP_SIZE
    lane_idx = tid % WARP_SIZE

    # ---- KV thread-to-data mapping (V2: warp w -> rows {w*2, w*2+1, w*2+16, w*2+17}) ----
    kv_ld_row_base = lane_idx / 32 * 16 + (lane_idx / 16) % 2 + warp_idx * 2
    kv_ld_col_base = _i32((lane_idx % 16) * 4)

    # ---- Resolve CSR slot -> physical KV row (PAGE_SIZE=1: phys == slot) ----
    def _get_kv_ld_row(kv_tile_start_i32, kv_tile_end_i32):
        row_idx = kv_ld_row_base + _idx(kv_tile_start_i32)
        phys_row = fx.Int32(-1)
        if row_idx < _idx(kv_tile_end_i32):
            phys_row = buffer_ops.buffer_load(indices_rsrc, row_idx, vec_width=1, dtype=T.i32)
        return _raw(phys_row)

    # ---- Async load one 32x64 KV block VRAM->LDS (zero-fill OOB rows) ----
    def _async_load_k_tile(p_lds_kv_warp, row_i32, col_base_i32, block_idx_const):
        lds_warp_offset = block_idx_const * KV_BLOCK_BYTES
        lds_adjust = lds_warp_offset - block_idx_const * KV_NUM_COLS
        lds_base_i32 = _i32(ArithValue(p_lds_kv_warp) + lds_adjust)

        is_oob = ArithValue(row_i32) == -1
        if is_oob:
            lds_addr = _i32(ArithValue(lds_base_i32) + block_idx_const * KV_NUM_COLS + _i32(lane_idx) * 4)
            _ptr_store(c_zero_i32, _lds_ptr_from_i32(lds_addr), alignment=4)
        else:
            voff = _i32(ArithValue(row_i32) * QK_HEAD_DIM + col_base_i32)
            rocdl.buffer_load_to_lds(
                kv_rsrc,
                _lds_ptr_from_i32(lds_base_i32),
                voff,
                offset=block_idx_const * KV_NUM_COLS,
            )

    def _async_load_kv_all(p_lds_kv_warp, row_i32, col_base_i32):
        for blk in range_constexpr(KV_NUM_BLOCKS):
            _async_load_k_tile(p_lds_kv_warp, row_i32, col_base_i32, blk)

    # ---- Per-lane K LDS base offset ----
    k_row_in_mfma = lane_idx % MFMA_M
    k_row_phy = (k_row_in_mfma / 2) * 4 + k_row_in_mfma % 2
    k_col_in_lane = (lane_idx / MFMA_M) * MFMA_ELEM_PER_THR
    k_lds_lane_offset = (
        (k_row_phy / 4) * KV_SUB_BYTES + (k_row_phy % 4) * KV_BYTES_PER_ROW + (k_col_in_lane % KV_NUM_COLS)
    )

    def _load_k_from_lds(k_base_i32, row_offset, col_offset):
        fixed_offset = (
            (row_offset // 16) * 2 * KV_BYTES_PER_ROW
            + (col_offset % KV_NUM_COLS)
            + (col_offset // KV_NUM_COLS) * KV_BLOCK_BYTES
        )
        return _lds_load_volatile(k_base_i32, T.i64, byte_offset=fixed_offset)

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
            data = _lds_load(lds_addr, T.i32x2, static_byte_offset=off)
            data_vec = Vec(data)
            v_vals.append(data_vec[0])
            v_vals.append(data_vec[1])
        return v_vals  # 8 i32

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

    def _store_vt_to_lds(vt_lds_base_idx, warp_idx_val, lane_idx_val, vt8):
        row_blk = (warp_idx_val % 2) * 4 + lane_idx_val / 16
        col_blk = (lane_idx_val % 16) + (warp_idx_val / 2) * 16
        block_offset = (row_blk * VT_BLKS_PER_ROW_PAD + col_blk) * VT_ELEMS_PER_BLK
        lds_vt_addr = vt_lds_base_idx + block_offset
        lo_packed = Vec.from_elements(vt8[0:4], fx.Int32)
        Vec(lo_packed).bitcast(fx.Int8).store(lds_buffer, [lds_vt_addr])
        hi_packed = Vec.from_elements(vt8[4:8], fx.Int32)
        Vec(hi_packed).bitcast(fx.Int8).store(lds_buffer, [lds_vt_addr + 16])

    def _load_vt_from_lds(vt_base_i32, col_offset):
        fixed_col_blk = col_offset // VT_COLS_PER_THR
        fixed_block_offset = fixed_col_blk * VT_ELEMS_PER_BLK
        offset_tl_bl = 4 * VT_BLKS_PER_ROW_PAD * VT_ELEMS_PER_BLK  # 8448
        v0 = _lds_load_volatile(vt_base_i32, T.i32, byte_offset=fixed_block_offset)
        v1 = _lds_load_volatile(vt_base_i32, T.i32, byte_offset=fixed_block_offset + offset_tl_bl)
        return v0, v1

    def _vt_base_i32():
        vt_row_blk = lane_idx / 16
        vt_col_blk = (lane_idx % 16) / VT_COLS_PER_THR
        vt_row_inblk = lane_idx % VT_ROWS_PER_THR
        vt_col_inblk = ((lane_idx % 8) / VT_ROWS_PER_THR) * VT_ROWS_PER_THR
        vt_block_offset = (vt_row_blk * VT_BLKS_PER_ROW_PAD + vt_col_blk) * VT_ELEMS_PER_BLK
        vt_inblock_offset = vt_row_inblk * VT_COLS_PER_THR + vt_col_inblk
        vt_lds_lane_offset = vt_block_offset + vt_inblock_offset
        return _i32(ArithValue(lds_base_idx + P_LDS_VT) + vt_lds_lane_offset)

    # ---- Warp reduce helpers ----
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

    # ---- Q loading: bf16 VRAM -> fp8 (cvt) -> LDS staging -> GPR ----
    def _bf16x4dw_to_fp8x2dw(i32x4_bf16):
        f = Vec(Vec(i32x4_bf16).bitcast(fx.BFloat16)).to(fx.Float32)  # 8 f32
        fr = [_raw(f[j]) for j in range(8)]
        w0 = rocdl.cvt_pk_fp8_f32(T.i32, fr[0], fr[1], c_zero_i32, 0)
        w0 = rocdl.cvt_pk_fp8_f32(T.i32, fr[2], fr[3], w0, 1)
        w1 = rocdl.cvt_pk_fp8_f32(T.i32, fr[4], fr[5], c_zero_i32, 0)
        w1 = rocdl.cvt_pk_fp8_f32(T.i32, fr[6], fr[7], w1, 1)
        return w0, w1

    def _load_q_to_regs(q_idx_val):
        p_lds_q_warp = lds_base_idx + P_LDS_Q + warp_idx * SZ_LDS_Q_PER_WARP

        row = lane_idx / 4
        col = (lane_idx % 4) * 16
        # bf16 element index into [num_queries, num_heads, head_dim]
        base_elem = (_idx(q_idx_val) * NUM_QO_HEADS + warp_idx * 16 + row) * QK_HEAD_DIM + col

        row_st = lane_idx / 4
        col_st = (lane_idx % 4) * 16
        lds_st_offset = (row_st / 2) * Q_BYTES_PER_2ROWS + (row_st % 2) * Q_ELEM_PER_ROW + col_st
        row_ld = lane_idx % 16
        col_ld = (lane_idx / 16) * 8
        lds_ld_offset = (row_ld / 2) * Q_BYTES_PER_2ROWS + (row_ld % 2) * Q_ELEM_PER_ROW + col_ld

        lds_st_addr = p_lds_q_warp + lds_st_offset
        lds_st_ptr = _inttoptr_lds(lds_st_addr)
        lds_rd_addr = p_lds_q_warp + lds_ld_offset

        q_regs = []
        for p in range_constexpr(Q_NUM_PASSES):
            # 16 bf16 cols (col..col+15) at column block p*64; i32-dword offset.
            elem = base_elem + p * Q_ELEM_PER_ROW
            dword_off = _i32(ArithValue(elem) // 2)
            lo = buffer_ops.buffer_load(query_rsrc, dword_off, vec_width=4, dtype=T.i32)  # 8 bf16
            hi = buffer_ops.buffer_load(query_rsrc, _raw(ArithValue(dword_off) + 4), vec_width=4, dtype=T.i32)
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
            w0a, w1a = _bf16x4dw_to_fp8x2dw(lo)
            w0b, w1b = _bf16x4dw_to_fp8x2dw(hi)
            q_fp8 = Vec.from_elements([w0a, w1a, w0b, w1b], fx.Int32)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            _ptr_store(q_fp8, lds_st_ptr, alignment=16)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            q0 = _lds_load(lds_rd_addr, T.i64, static_byte_offset=0)
            q1 = _lds_load(lds_rd_addr, T.i64, static_byte_offset=MFMA_K)
            q_regs.append((q0, q1))

        q_nope_packs = []
        for p in range_constexpr(Q_NUM_PASSES):
            q_nope_packs.append(q_regs[p][0])
            q_nope_packs.append(q_regs[p][1])
        return q_nope_packs  # 16 i64

    # ---- Softmax scale + CSR/boundary masking ----
    def _softmax_scale_p(p_vals, col_0_start, kv_end_i32):
        result = [None] * P_VALS_PER_THR
        for i in range_constexpr(P_VALS_PER_THR):
            result[i] = _f32(p_vals[i]) * softmax_scale
        kv_end = _idx(kv_end_i32)
        skv = ArithValue(num_kv_rows)
        for i in range_constexpr(P_VALS_PER_THR):
            sub_offset = (i // 4) * 16 + (i % 4)
            pos = col_0_start + sub_offset
            slot = buffer_ops.buffer_load(indices_rsrc, _i32(pos), vec_width=1, dtype=T.i32)
            slot_a = ArithValue(slot)
            inv = ArithValue(_raw(pos >= kv_end))
            inv = inv | (slot_a < 0)
            inv = inv | (slot_a >= skv)
            result[i] = ArithValue(_raw(inv)).select(_raw(c_neg_inf), result[i])
        return result

    def _softmax(p_vals, row_max_old, row_sum_e_old, is_first_iter, kv_tile_start_i32, kv_end_i32):
        col_0_start = lane_idx / 16 * 4 + _idx(kv_tile_start_i32)
        scaled = _softmax_scale_p(p_vals, col_0_start, kv_end_i32)

        local_max = scaled[0]
        for i in range_constexpr(1, P_VALS_PER_THR):
            local_max = _fmax(local_max, scaled[i])
        local_max = _warp_reduce_max_16(local_max)

        if const_expr(is_first_iter):
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

        if const_expr(is_first_iter):
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

    # ---- GEMM1 + softmax + V-transpose for one KV tile ----
    def _process_tile_gemm1(p_lds_kv_base, kv_tile_start_i32, kv_end_i32, q_nope, rm_in, rse_in, is_first):
        k_base_i32 = _i32(ArithValue(p_lds_kv_base) + k_lds_lane_offset)
        P_COMP_SUBS = BLOCK_N // MFMA_N  # 2
        p_comp = [c_zero_v4f32] * P_COMP_SUBS

        for nope_pair in range_constexpr(NUM_NOPE_ITERS):
            tile_0 = nope_pair * 2
            tile_1 = nope_pair * 2 + 1
            k0 = [_load_k_from_lds(k_base_i32, 16 * h, tile_0 * BLOCK_K) for h in range_constexpr(P_COMP_SUBS)]
            k1 = [_load_k_from_lds(k_base_i32, 16 * h, tile_1 * BLOCK_K) for h in range_constexpr(P_COMP_SUBS)]
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

        p_vals = []
        for sub in range_constexpr(P_COMP_SUBS):
            pv = Vec(p_comp[sub])
            for ii in range_constexpr(4):
                p_vals.append(pv[ii])

        v8_raw = _load_v_from_lds(p_lds_kv_base, warp_idx, lane_idx)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        rocdl.sched_barrier(0)

        p_exp_vals, rm_new, rse_new, rescale = _softmax(
            p_vals, rm_in, rse_in, is_first, kv_tile_start_i32, kv_end_i32
        )
        p_pack = _pack_p_to_fp8(p_exp_vals)
        vt8 = _transpose_v(v8_raw)
        _store_vt_to_lds(lds_base_idx + P_LDS_VT, warp_idx, lane_idx, vt8)
        return rm_new, rse_new, p_pack, rescale

    # ---- GEMM2: P @ V accumulation (K_HALVES == 1) ----
    def _gemm2_core(p_pack, oaccu, vt_base_i32):
        for pv_pair in range_constexpr(NUM_PV_ITERS // 2):
            iter_a = pv_pair * 2
            iter_b = pv_pair * 2 + 1
            col_a_strip = iter_a * MFMA_N * 2
            col_b_strip = iter_b * MFMA_N * 2

            vta0_lo, vta0_hi = _load_vt_from_lds(vt_base_i32, col_a_strip)
            vta1_lo, vta1_hi = _load_vt_from_lds(vt_base_i32, col_a_strip + MFMA_N)
            vtb0_lo, vtb0_hi = _load_vt_from_lds(vt_base_i32, col_b_strip)
            vtb1_lo, vtb1_hi = _load_vt_from_lds(vt_base_i32, col_b_strip + MFMA_N)

            read0_lo = [vta0_lo, vtb0_lo]
            read0_hi = [vta0_hi, vtb0_hi]
            read1_lo = [vta1_lo, vtb1_lo]
            read1_hi = [vta1_hi, vtb1_hi]
            iter_idxs = [iter_a, iter_b]
            wait_lgkm = [4, 0]

            for step in range_constexpr(2):
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=wait_lgkm[step]))
                lhs0 = _pack_i32x2(read0_lo[step], read0_hi[step])
                lhs1 = _pack_i32x2(read1_lo[step], read1_hi[step])
                iter_idx = iter_idxs[step]
                acc_idx = iter_idx * 2
                oaccu[acc_idx] = _mfma_fp8(T.f32x4, [lhs0, p_pack, oaccu[acc_idx], 0, 0, 0])
                oaccu[acc_idx + 1] = _mfma_fp8(T.f32x4, [lhs1, p_pack, oaccu[acc_idx + 1], 0, 0, 0])
            rocdl.sched_barrier(0)
        return oaccu

    def _gemm2_first_iter(p_pack, vt_base_i32):
        _barrier(lgkmcnt=0)
        rocdl.sched_barrier(0)
        oaccu = [c_zero_v4f32] * (NUM_PV_ITERS * 2)
        return _gemm2_core(p_pack, oaccu, vt_base_i32)

    def _gemm2_with_rescale(p_pack, rescale, oaccu_in, vt_base_i32):
        oaccu = _rescale_oaccu(oaccu_in, rescale)
        _barrier(lgkmcnt=0)
        rocdl.sched_barrier(0)
        return _gemm2_core(p_pack, oaccu, vt_base_i32)

    # ---- bf16 output store via LDS reshape ----
    def _pack_f32x4_to_bf16_2dw(acc_val):
        i16s = Vec(acc_val).to(fx.BFloat16).bitcast(fx.Int16)
        i16_0, i16_1, i16_2, i16_3 = (_raw(i16s[j]) for j in range(4))
        dw0 = _raw(ArithValue(i16_0).extui(T.i32) | (ArithValue(i16_1).extui(T.i32) << 16))
        dw1 = _raw(ArithValue(i16_2).extui(T.i32) | (ArithValue(i16_3).extui(T.i32) << 16))
        return dw0, dw1

    def _store_oaccu_pair_bf16(oaccu_a, oaccu_b, tile_idx, p_lds_o, row_base_i32):
        o16_row_st = lane_idx % 16
        o16_col_st = (lane_idx / 16) * 4
        o16_st_offset = _raw(
            ((o16_row_st / 2) * O16_ELEM_PER_PAD_2ROWS + (o16_row_st % 2) * O16_NUM_COLS + o16_col_st) * 2
        )
        o16_row_ld = lane_idx / 4
        o16_col_ld = (lane_idx % 4) * 8
        o16_rd_offset = _raw(
            ((o16_row_ld / 2) * O16_ELEM_PER_PAD_2ROWS + (o16_row_ld % 2) * O16_NUM_COLS + o16_col_ld) * 2
        )
        lds_warp = ArithValue(p_lds_o) + warp_idx * O16_LDS_PER_WARP
        lds_st_addr = _i32(ArithValue(lds_warp) + o16_st_offset)

        for sub, acc_val in enumerate([oaccu_a, oaccu_b]):
            dw0, dw1 = _pack_f32x4_to_bf16_2dw(acc_val)
            vec_2dw = Vec.from_elements([dw0, dw1], fx.Int32)
            st_addr_sub = _i32(ArithValue(lds_st_addr) + sub * O16_NUM_COLS)
            _ptr_store(vec_2dw, _lds_ptr_from_i32(st_addr_sub), alignment=8, volatile_=True)

        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        lds_rd_addr = _i32(ArithValue(lds_warp) + o16_rd_offset)
        data = _ptr_load(T.i32x4, _lds_ptr_from_i32(lds_rd_addr), alignment=16)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

        row_vram = ArithValue(row_base_i32) + o16_row_ld
        col_vram = ArithValue(o16_col_ld) + tile_idx * MFMA_N * 2
        vram_offset = _raw((row_vram * V_HEAD_DIM + col_vram) * 2)
        buffer_ops.buffer_store(data, final_output_rsrc, vram_offset, offset_is_bytes=True)

    def _normalize_and_store(oaccu, rse, valid_i1, row_base_idx):
        p_lds_o = p_lds_kv_0_base
        reci = rocdl.rcp(T.f32, _raw(rse))
        reci_vec = _raw(Vec.filled(4, fx.Float32(reci), fx.Float32))
        _barrier(lgkmcnt=0)
        for pv in range_constexpr(NUM_PV_ITERS):
            a0 = _f32(oaccu[pv * 2]) * reci_vec
            a1 = _f32(oaccu[pv * 2 + 1]) * reci_vec
            a0 = ArithValue(_raw(valid_i1)).select(_raw(a0), _raw(c_zero_v4f32))
            a1 = ArithValue(_raw(valid_i1)).select(_raw(a1), _raw(c_zero_v4f32))
            _store_oaccu_pair_bf16(a0, a1, pv, p_lds_o, row_base_idx)

    # ==================================================================
    # KV LDS buffer pointers
    # ==================================================================
    p_lds_kv_0_base = lds_base_idx + P_LDS_KV_0

    def _kv_warp_lds_base(p_lds_kv_base):
        warp_offset = _raw(ArithValue(_uniform_i32(warp_idx)) * KV_SUB_BYTES)
        return _raw(ArithValue(_i32(p_lds_kv_base)) + warp_offset)

    p_lds_kv_0_warp = _kv_warp_lds_base(p_lds_kv_0_base)

    # ==================================================================
    # Per-query attention
    # ==================================================================
    # CSR range for this query token.
    rng = buffer_ops.buffer_load(indptr_rsrc, q_idx, vec_width=2, dtype=T.i32)
    rng_vec = Vec(rng)
    kv_start = rocdl.readfirstlane(T.i32, rng_vec[0])
    kv_end = rocdl.readfirstlane(T.i32, rng_vec[1])
    kv_start_v = ArithValue(kv_start)
    kv_len = _raw(ArithValue(kv_end) - kv_start_v)

    row_base = _idx(q_idx) * NUM_QO_HEADS + warp_idx * 16

    # Load Q (bf16 -> fp8) unconditionally; cheap relative to KV path.
    q_nope_packs = _load_q_to_regs(q_idx)

    num_tiles = (ArithValue(kv_len) + (BLOCK_N - 1)).with_signedness(False) // BLOCK_N
    has_multi_tiles = ArithValue(kv_len) > BLOCK_N

    def _attend_first_tile():
        row0 = _get_kv_ld_row(kv_start, kv_end)
        _async_load_kv_all(p_lds_kv_0_warp, row0, kv_ld_col_base)
        _barrier(vmcnt=0, lgkmcnt=0)
        rocdl.sched_barrier(0)
        rm0, rse0, p_pack0, _resc0 = _process_tile_gemm1(
            p_lds_kv_0_base, kv_start, kv_end, q_nope_packs, c_neg_inf, c_zero_f32, True
        )
        oaccu0 = _gemm2_first_iter(p_pack0, _vt_base_i32())
        return rm0, rse0, oaccu0

    def _attend_tile(kv_tile_start_i32, rm_in, rse_in, oaccu_in):
        row = _get_kv_ld_row(kv_tile_start_i32, kv_end)
        _barrier(vmcnt=0, lgkmcnt=0)
        _async_load_kv_all(p_lds_kv_0_warp, row, kv_ld_col_base)
        _barrier(vmcnt=0, lgkmcnt=0)
        rocdl.sched_barrier(0)
        rm_n, rse_n, p_pack, rescale = _process_tile_gemm1(
            p_lds_kv_0_base, kv_tile_start_i32, kv_end, q_nope_packs, rm_in, rse_in, False
        )
        oaccu_n = _gemm2_with_rescale(p_pack, rescale, oaccu_in, _vt_base_i32())
        return rm_n, rse_n, oaccu_n

    # First tile always runs (kv_len==0 -> fully masked -> zero output).
    rm_first, rse_first, oaccu_first = _attend_first_tile()

    # The yield-driven loop must live inside its own helper (mirrors the MLA
    # decode template). Placing the ``yield`` directly inside the ``@flyc.jit``
    # function -- alongside the single-tile else branch -- breaks the structured
    # loop lowering and produces data-independent garbage.
    def _multi_tile_path():
        init_args = [rm_first, rse_first] + oaccu_first
        for tile_iv, state in range(_idx(1), _idx(num_tiles), _idx(1), init=init_args):
            tile_iv_i32 = ArithValue(fx.Int32(tile_iv))
            kv_tile_start_i32 = _raw(kv_start_v + tile_iv_i32 * BLOCK_N)
            rm_c = state[0]
            rse_c = state[1]
            oaccu_c = [state[2 + i] for i in range(NUM_PV_ITERS * 2)]
            rm_n, rse_n, oaccu_n = _attend_tile(kv_tile_start_i32, rm_c, rse_c, oaccu_c)
            results = yield [rm_n, rse_n] + oaccu_n
        rse_final = results[1]
        oaccu_final = [results[2 + i] for i in range(NUM_PV_ITERS * 2)]
        valid = _raw(ArithValue(_raw(rse_final)) > c_zero_f32)
        _normalize_and_store(oaccu_final, rse_final, valid, row_base)

    def _single_tile_path():
        valid = _raw(ArithValue(_raw(rse_first)) > c_zero_f32)
        _normalize_and_store(oaccu_first, rse_first, valid, row_base)

    @flyc.jit
    def _dispatch():
        if has_multi_tiles:
            _multi_tile_path()
        else:
            _single_tile_path()

    _dispatch()


# ---------------------------------------------------------------------------
# JIT launcher
# ---------------------------------------------------------------------------
@flyc.jit
def launch_sparse_mla_prefill(
    query: fx.Tensor,
    kv_buffer: fx.Tensor,
    indices: fx.Tensor,
    indptr: fx.Tensor,
    final_output: fx.Tensor,
    softmax_scale: fx.Float32,
    num_queries: fx.Int32,
    num_kv_rows: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = arith.index_cast(T.index, _raw(num_queries))
    kn_sparse_mla_prefill(
        query,
        kv_buffer,
        indices,
        indptr,
        final_output,
        softmax_scale,
        num_kv_rows,
    ).launch(
        grid=(grid_x, 1, 1),
        block=(NUM_THREADS, 1, 1),
        smem=0,
        stream=stream,
    )


# ===========================================================================
# Phase B: paged fp8_ds_mla kernel (single/two region, sink, UE8M0, paging)
#
# Strategy (docs/sparse-mla-prefill): reuse the Phase A flat-512 fp8 MFMA /
# online-softmax / software V-transpose machinery verbatim by materialising a
# 512-wide *fnuz* fp8 KV tile in LDS during the load:
#   - NoPE (448, blocks 0..6): region0 (SWA, fnuz) streamed VRAM->LDS by DMA;
#     region1 (compressed, OCP) + any UE8M0 != 1 go through a register-staged
#     CONVERT path that decodes->scales(power-of-2)->re-encodes fnuz, baking the
#     OCP x2 exponent correction and the UE8M0 per-64-block scale into the fp8
#     bytes (lossless for in-range power-of-2 shifts).
#   - RoPE (64, block 7): the bf16 cache tail is re-quantised to fnuz fp8.
#   - Q (512): bf16 -> fnuz fp8 in-kernel (unchanged from Phase A).
# Two regions share one (m, l, acc) online-softmax state (main tiles then extra
# tiles, matching the Triton decode kernel). Sink folds into the denominator.
# ===========================================================================
def compile_sparse_mla_prefill_paged(
    *,
    num_regions: int = 1,
    has_sink: bool = False,
    r0_convert: bool = False,
    r0_is_ocp: bool = False,
    r1_is_ocp: bool = True,
    waves_per_eu: int = 2,
    softmax_scale: float | None = None,
    single_request: bool = True,
    head_dim: int = 512,
    v_dim: int = 512,
    cache_layout: str = "fp8_ds_mla",
    scale_mode: str = "none",
    block_n: int = 32,
):
    """Build a paged sparse MLA prefill launcher (gfx942).

    num_regions: 1 (B1 SWA-only / GLM) or 2 (B2 compressed + SWA).
    has_sink:    fold a per-head virtual key into the softmax denominator.
    r0_convert:  region0 uses the register-staged convert load (needed when
                 UE8M0 != 1 or region0 is OCP); else the fast DMA path.
    r0_is_ocp / r1_is_ocp: per-region NoPE fp8 convention (fnuz vs OCP).
    head_dim:    512 (DSv4: 448 nope + 64 rope) or 576 (GLM/DSv3.2:
                 512 latent + 64 rope).  v_dim is always 512.
    cache_layout:
       "fp8_ds_mla" -- DSv4 584-byte packed rows (448 fp8 nope + 128 B bf16
                       rope + 8 B UE8M0 scale region).
       "glm_flat576" -- GLM/DSv3.2 flat fp8 rows (``head_dim`` fp8 bytes per
                       token, latent+rope both fp8, per-tensor scale, no scale
                       region).  Single region only.
    scale_mode:  ``per_tensor`` enables runtime ``q_scale`` / ``kv_scale`` f32
                 launch args (GLM).  DSv4 UE8M0 scales stay in the cache bytes.
    """
    NREG = int(num_regions)
    HAS_SINK = bool(has_sink)
    R0_CONVERT = bool(r0_convert)
    R0_OCP = bool(r0_is_ocp)
    R1_OCP = bool(r1_is_ocp)
    SINGLE_REQUEST = bool(single_request)

    # ---- head-dim-dependent constants (shadow the module 512 defaults) ----
    GLM_FLAT = cache_layout == "glm_flat576"
    USE_PT_SCALE = scale_mode == "per_tensor"
    if cache_layout not in ("fp8_ds_mla", "glm_flat576"):
        raise NotImplementedError(f"unknown cache_layout {cache_layout!r}")
    if GLM_FLAT and NREG != 1:
        raise NotImplementedError("glm_flat576 is single-region only")
    if GLM_FLAT and not USE_PT_SCALE:
        raise NotImplementedError("glm_flat576 requires scale_mode='per_tensor'")
    QK_HEAD_DIM = int(head_dim)
    V_HEAD_DIM = int(v_dim)
    if V_HEAD_DIM != 512:
        raise NotImplementedError(f"v_dim must be 512, got {V_HEAD_DIM}")
    if QK_HEAD_DIM % KV_NUM_COLS != 0 or QK_HEAD_DIM % Q_ELEM_PER_ROW != 0:
        raise NotImplementedError(f"head_dim must be a multiple of 64, got {QK_HEAD_DIM}")

    # ---- BLOCK_N (KV tile rows) as a compile-time constant ----
    # All KV / V^T / softmax LDS constants below shadow the module-level (=32)
    # defaults so the kernel body (a closure) picks them up.  The gfx942 V2
    # software-transpose KV layout (KvManagerV2 + VtManagerV1) is hardwired to
    # KV_ROWS_PER_SUB == 4 (i.e. BLOCK_N == 32): the per-warp row mapping, the
    # K/V LDS readers, and the 4x8 register transpose all assume 4 rows/sub
    # split across two 16-row MFMA sub-tiles.  Other BLOCK_N values are gated
    # below (see docs/issues/sparse-mla-prefill-blockn).
    BLOCK_N = int(block_n)
    if BLOCK_N % NUM_WARPS != 0:
        raise NotImplementedError(f"block_n must be divisible by NUM_WARPS={NUM_WARPS}, got {BLOCK_N}")
    if BLOCK_N % MFMA_N != 0:
        raise NotImplementedError(f"block_n must be divisible by MFMA_N={MFMA_N}, got {BLOCK_N}")
    KV_ROWS_PER_SUB = BLOCK_N // NUM_WARPS
    KV_NUM_SUBS = BLOCK_N // KV_ROWS_PER_SUB  # == NUM_WARPS
    KV_SUB_BYTES = KV_ROWS_PER_SUB * KV_BYTES_PER_ROW + KV_PAD_DW * 4
    KV_BLOCK_BYTES = KV_SUB_BYTES * KV_NUM_SUBS
    P_VALS_PER_THR = (BLOCK_N * MFMA_M) // WARP_SIZE
    SZ_LDS_VT = VT_NUM_SUB_BLKS * ((BLOCK_N // VT_NUM_SUB_BLKS) * V_HEAD_DIM + 16 * 4)
    P_LDS_VT = 0
    P_LDS_Q = SZ_LDS_VT

    KV_NUM_BLOCKS = QK_HEAD_DIM // KV_NUM_COLS  # 8 (512) or 9 (576)
    SZ_LDS_KV = KV_BLOCK_BYTES * KV_NUM_BLOCKS
    P_LDS_KV_0 = P_LDS_Q + SZ_LDS_Q
    TOTAL_LDS_BYTES = P_LDS_KV_0 + SZ_LDS_KV
    # BLOCK_N=64 @ head_dim=576 lands here (TOTAL_LDS_BYTES=79424 > 65536).
    if TOTAL_LDS_BYTES > 65536:
        raise NotImplementedError(
            f"block_n={BLOCK_N} needs {TOTAL_LDS_BYTES} B LDS > 65536 B gfx942 cap "
            f"(head_dim={QK_HEAD_DIM}); larger tiles require gfx950 HW V-transpose."
        )
    assert SZ_LDS_O16 <= SZ_LDS_KV, "Output LDS must fit in the KV buffer region"
    # The V2 software-transpose data layout is only correct for 4 rows/sub.
    if KV_ROWS_PER_SUB != 4:
        raise NotImplementedError(
            f"block_n={BLOCK_N} (KV_ROWS_PER_SUB={KV_ROWS_PER_SUB}) is unsupported on the "
            "gfx942 V2 software-transpose path, which hardwires 4 rows/sub (BLOCK_N=32). "
            "BLOCK_N=16 needs a KvManagerV2 row-mapping + V-transpose redesign; "
            "BLOCK_N=64 needs gfx950 ds_read_b64_tr_b8. "
            "See docs/issues/sparse-mla-prefill-blockn."
        )
    Q_NUM_PASSES = QK_HEAD_DIM // Q_ELEM_PER_ROW
    NUM_NOPE_ITERS = QK_HEAD_DIM // (MFMA_K * 2)
    # Flat (GLM) rows have no scale region and store rope as fp8, so the row
    # stride equals head_dim bytes; DSv4 packed rows are 584 B with a 576 B
    # data sub-region.
    PK_TOKEN_BYTES = QK_HEAD_DIM if GLM_FLAT else 576
    PK_CACHE_ROW = QK_HEAD_DIM if GLM_FLAT else 584

    base_scale = (QK_HEAD_DIM ** -0.5) if softmax_scale is None else float(softmax_scale)
    SOFTMAX_SCALE = fx.Float32(base_scale)

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def kn_sparse_mla_prefill_paged(
        query: fx.Tensor,            # [nq*128, 512] bf16
        main_cache: fx.Tensor,       # uint8 packed fp8_ds_mla
        main_indices: fx.Tensor,     # i32 CSR values (slot ids)
        main_indptr: fx.Tensor,      # i32 [nq+1]
        main_block_table: fx.Tensor, # i32 [num_reqs*max_blocks]
        extra_cache: fx.Tensor,      # uint8 (dummy if NREG==1)
        extra_indices: fx.Tensor,
        extra_indptr: fx.Tensor,
        extra_block_table: fx.Tensor,
        q_req: fx.Tensor,            # i32 [nq] query -> request id (ignored if SINGLE_REQUEST)
        sink_buf: fx.Tensor,         # f32 [128] (ignored if not HAS_SINK)
        final_output: fx.Tensor,     # [nq*128, 512] bf16
        q_scale_buf: fx.Tensor,      # f32 [1] (ignored unless USE_PT_SCALE)
        kv_scale_buf: fx.Tensor,     # f32 [1] (ignored unless USE_PT_SCALE)
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
        if const_expr(USE_PT_SCALE):
            q_scale_rsrc = buffer_ops.create_buffer_resource(q_scale_buf)
            kv_scale_rsrc = buffer_ops.create_buffer_resource(kv_scale_buf)
            q_sc = _f32(
                rocdl.readfirstlane(
                    T.f32,
                    buffer_ops.buffer_load(q_scale_rsrc, c_zero_i32, vec_width=1, dtype=T.f32),
                )
            )
            kv_sc = _f32(
                rocdl.readfirstlane(
                    T.f32,
                    buffer_ops.buffer_load(kv_scale_rsrc, c_zero_i32, vec_width=1, dtype=T.f32),
                )
            )
            q_sc = _fmax(q_sc, fx.Float32(1e-30))
            q_sc_inv = _f32(rocdl.rcp(T.f32, _raw(q_sc)))
            qk_softmax_scale = _fmul(_fmul(softmax_scale, q_sc), kv_sc)
        else:
            qk_softmax_scale = softmax_scale

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
        def _row_addrs(idx_rsrc, bt_rsrc, num_rows_i32, block_size_i32, max_blocks_i32, kv_tile_start_i32, kv_end_i32):
            row_idx = kv_ld_row_base + _idx(kv_tile_start_i32)
            in_range = row_idx < _idx(kv_end_i32)
            slot = buffer_ops.buffer_load(idx_rsrc, _i32(row_idx), vec_width=1, dtype=T.i32)
            slot_a = ArithValue(slot)
            valid = ArithValue(_raw(in_range)) & (slot_a >= 0) & (slot_a < ArithValue(num_rows_i32))
            safe_slot = ArithValue(_raw(valid)).select(_raw(slot), _raw(c_zero_i32))
            bsz = ArithValue(block_size_i32)
            block_idx = ArithValue(safe_slot).with_signedness(False) // bsz
            pos = ArithValue(safe_slot) - ArithValue(_raw(block_idx)) * bsz
            bt_index = ArithValue(req_id) * ArithValue(max_blocks_i32) + ArithValue(_raw(block_idx))
            phys = buffer_ops.buffer_load(bt_rsrc, _i32(bt_index), vec_width=1, dtype=T.i32)
            blk_stride = bsz * PK_CACHE_ROW
            token_base = ArithValue(phys) * ArithValue(_raw(blk_stride)) + ArithValue(_raw(pos)) * PK_TOKEN_BYTES
            scale_base = (
                ArithValue(phys) * ArithValue(_raw(blk_stride))
                + bsz * PK_TOKEN_BYTES
                + ArithValue(_raw(pos)) * 8
            )
            tb = ArithValue(_raw(valid)).select(_raw(token_base), _raw(fx.Int32(-1)))
            sb = ArithValue(_raw(valid)).select(_raw(scale_base), _raw(c_zero_i32))
            return _i32(tb), _i32(sb)

        # ---- NoPE DMA load (region0 fnuz fast path), blocks 0..6 ----
        def _load_nope_dma(cache_rsrc, p_lds_kv_warp, token_base_i32):
            for blk in range_constexpr(PK_NOPE_BLOCKS):
                lds_adjust = blk * KV_BLOCK_BYTES - blk * KV_NUM_COLS
                lds_base_i32 = _i32(ArithValue(p_lds_kv_warp) + lds_adjust)
                is_oob = ArithValue(token_base_i32) == -1
                if is_oob:
                    lds_addr = _i32(ArithValue(lds_base_i32) + blk * KV_NUM_COLS + _i32(lane_idx) * 4)
                    _ptr_store(c_zero_i32, _lds_ptr_from_i32(lds_addr), alignment=4)
                else:
                    voff = _i32(ArithValue(token_base_i32) + kv_ld_col_base)
                    rocdl.buffer_load_to_lds(
                        cache_rsrc, _lds_ptr_from_i32(lds_base_i32), voff, offset=blk * KV_NUM_COLS
                    )

        # Flush NaN to 0 in the integer domain (immune to nnan fastmath). OCP
        # byte 0x80 is -0.0, but cvt_pk_f32_fp8 decodes fp8 as e4m3*fnuz*, where
        # 0x80 is NaN. -0.0 -> 0.0 is the correct value, so flushing is exact for
        # the only NaN-producing OCP code that appears in real KV data.
        def _flush_nan(fval):
            bits = _raw(ArithValue(_f32(fval)).bitcast(T.i32))
            absb = ArithValue(bits) & 0x7FFFFFFF
            is_nan = _raw(absb > 0x7F800000)
            return ArithValue(is_nan).select(_raw(c_zero_f32), _raw(_f32(fval)))

        # ---- NoPE convert load (OCP and/or UE8M0), blocks 0..6 ----
        # ``bias_f32`` is the OCP->fnuz exponent correction (1.0 for OCP bytes,
        # 0.0 for fnuz) added on top of the UE8M0 (enc-127) per-block exponent.
        # It may be a compile-time constant (B1) or a runtime select (B2).
        def _load_nope_convert(cache_rsrc, p_lds_kv_warp, token_base_i32, scale_base_i32, bias_f32):
            for blk in range_constexpr(PK_NOPE_BLOCKS):
                dst_addr = _i32(ArithValue(p_lds_kv_warp) + blk * KV_BLOCK_BYTES + _i32(lane_idx) * 4)
                is_oob = ArithValue(token_base_i32) == -1
                if is_oob:
                    _ptr_store(c_zero_i32, _lds_ptr_from_i32(dst_addr), alignment=4)
                else:
                    byte_off = ArithValue(token_base_i32) + kv_ld_col_base + blk * KV_NUM_COLS
                    word = buffer_ops.buffer_load(
                        cache_rsrc, _i32(byte_off.with_signedness(False) // 4), vec_width=1, dtype=T.i32
                    )
                    f01 = Vec(rocdl.cvt_pk_f32_fp8(T.f32x2, _raw(word), 0))
                    f23 = Vec(rocdl.cvt_pk_f32_fp8(T.f32x2, _raw(word), 1))
                    # UE8M0 per-64-block exponent byte (+ OCP x2 bias)
                    s_byte_off = ArithValue(scale_base_i32) + blk
                    s_word = buffer_ops.buffer_load(
                        cache_rsrc, _i32(s_byte_off.with_signedness(False) // 4), vec_width=1, dtype=T.i32
                    )
                    s_shift = (ArithValue(s_byte_off) & 3) * 8
                    enc = (ArithValue(_raw(s_word)).with_signedness(False) >> ArithValue(_raw(s_shift))) & 0xFF
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
        def _load_rope_block(cache_rsrc, p_lds_kv_warp, token_base_i32):
            dst_addr = _i32(ArithValue(p_lds_kv_warp) + PK_ROPE_BLOCK * KV_BLOCK_BYTES + _i32(lane_idx) * 4)
            is_oob = ArithValue(token_base_i32) == -1
            if is_oob:
                _ptr_store(c_zero_i32, _lds_ptr_from_i32(dst_addr), alignment=4)
            else:
                byte_off = ArithValue(token_base_i32) + PK_NOPE_BYTES + kv_ld_col_base * 2
                pair = buffer_ops.buffer_load(
                    cache_rsrc, _i32(byte_off.with_signedness(False) // 4), vec_width=2, dtype=T.i32
                )
                bf = Vec(Vec(pair).bitcast(fx.BFloat16)).to(fx.Float32)  # 4 f32
                w = rocdl.cvt_pk_fp8_f32(T.i32, _raw(bf[0]), _raw(bf[1]), c_zero_i32, 0)
                w = rocdl.cvt_pk_fp8_f32(T.i32, _raw(bf[2]), _raw(bf[3]), w, 1)
                _ptr_store(w, _lds_ptr_from_i32(dst_addr), alignment=4)

        # ---- GLM flat fp8 load: whole row is fp8 (latent+rope), row stride =
        # head_dim bytes; DMA all KV_NUM_BLOCKS 64-col blocks straight to LDS.
        # The per-tensor scale is folded in the softmax/output, not here.
        def _load_flat_dma(cache_rsrc, p_lds_kv_warp, token_base_i32):
            for blk in range_constexpr(KV_NUM_BLOCKS):
                lds_adjust = blk * KV_BLOCK_BYTES - blk * KV_NUM_COLS
                lds_base_i32 = _i32(ArithValue(p_lds_kv_warp) + lds_adjust)
                is_oob = ArithValue(token_base_i32) == -1
                if is_oob:
                    lds_addr = _i32(ArithValue(lds_base_i32) + blk * KV_NUM_COLS + _i32(lane_idx) * 4)
                    _ptr_store(c_zero_i32, _lds_ptr_from_i32(lds_addr), alignment=4)
                else:
                    voff = _i32(ArithValue(token_base_i32) + kv_ld_col_base)
                    rocdl.buffer_load_to_lds(
                        cache_rsrc, _lds_ptr_from_i32(lds_base_i32), voff, offset=blk * KV_NUM_COLS
                    )

        # ---- K/V LDS readers (identical to Phase A KvManagerV2 layout) ----
        k_row_in_mfma = lane_idx % MFMA_M
        k_row_phy = (k_row_in_mfma / 2) * 4 + k_row_in_mfma % 2
        k_col_in_lane = (lane_idx / MFMA_M) * MFMA_ELEM_PER_THR
        k_lds_lane_offset = (
            (k_row_phy / 4) * KV_SUB_BYTES + (k_row_phy % 4) * KV_BYTES_PER_ROW + (k_col_in_lane % KV_NUM_COLS)
        )

        def _load_k_from_lds(k_base_i32, row_offset, col_offset):
            fixed_offset = (
                (row_offset // 16) * 2 * KV_BYTES_PER_ROW
                + (col_offset % KV_NUM_COLS)
                + (col_offset // KV_NUM_COLS) * KV_BLOCK_BYTES
            )
            return _lds_load_volatile(k_base_i32, T.i64, byte_offset=fixed_offset)

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
                data = _lds_load(lds_addr, T.i32x2, static_byte_offset=off)
                data_vec = Vec(data)
                v_vals.append(data_vec[0])
                v_vals.append(data_vec[1])
            return v_vals

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

        def _store_vt_to_lds(vt_lds_base_idx, warp_idx_val, lane_idx_val, vt8):
            row_blk = (warp_idx_val % 2) * 4 + lane_idx_val / 16
            col_blk = (lane_idx_val % 16) + (warp_idx_val / 2) * 16
            block_offset = (row_blk * VT_BLKS_PER_ROW_PAD + col_blk) * VT_ELEMS_PER_BLK
            lds_vt_addr = vt_lds_base_idx + block_offset
            lo_packed = Vec.from_elements(vt8[0:4], fx.Int32)
            Vec(lo_packed).bitcast(fx.Int8).store(lds_buffer, [lds_vt_addr])
            hi_packed = Vec.from_elements(vt8[4:8], fx.Int32)
            Vec(hi_packed).bitcast(fx.Int8).store(lds_buffer, [lds_vt_addr + 16])

        def _load_vt_from_lds(vt_base_i32, col_offset):
            fixed_col_blk = col_offset // VT_COLS_PER_THR
            fixed_block_offset = fixed_col_blk * VT_ELEMS_PER_BLK
            offset_tl_bl = 4 * VT_BLKS_PER_ROW_PAD * VT_ELEMS_PER_BLK
            v0 = _lds_load_volatile(vt_base_i32, T.i32, byte_offset=fixed_block_offset)
            v1 = _lds_load_volatile(vt_base_i32, T.i32, byte_offset=fixed_block_offset + offset_tl_bl)
            return v0, v1

        def _vt_base_i32():
            vt_row_blk = lane_idx / 16
            vt_col_blk = (lane_idx % 16) / VT_COLS_PER_THR
            vt_row_inblk = lane_idx % VT_ROWS_PER_THR
            vt_col_inblk = ((lane_idx % 8) / VT_ROWS_PER_THR) * VT_ROWS_PER_THR
            vt_block_offset = (vt_row_blk * VT_BLKS_PER_ROW_PAD + vt_col_blk) * VT_ELEMS_PER_BLK
            vt_inblock_offset = vt_row_inblk * VT_COLS_PER_THR + vt_col_inblk
            vt_lds_lane_offset = vt_block_offset + vt_inblock_offset
            return _i32(ArithValue(lds_base_idx + P_LDS_VT) + vt_lds_lane_offset)

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
            fr = []
            for j in range_constexpr(8):
                if const_expr(USE_PT_SCALE):
                    fr.append(_fmul(_f32(f[j]), q_sc_inv))
                else:
                    fr.append(_raw(f[j]))
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, fr[0], fr[1], c_zero_i32, 0)
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, fr[2], fr[3], w0, 1)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, fr[4], fr[5], c_zero_i32, 0)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, fr[6], fr[7], w1, 1)
            return w0, w1

        def _load_q_to_regs(q_idx_val):
            p_lds_q_warp = lds_base_idx + P_LDS_Q + warp_idx * SZ_LDS_Q_PER_WARP
            row = lane_idx / 4
            col = (lane_idx % 4) * 16
            base_elem = (_idx(q_idx_val) * NUM_QO_HEADS + warp_idx * 16 + row) * QK_HEAD_DIM + col
            row_st = lane_idx / 4
            col_st = (lane_idx % 4) * 16
            lds_st_offset = (row_st / 2) * Q_BYTES_PER_2ROWS + (row_st % 2) * Q_ELEM_PER_ROW + col_st
            row_ld = lane_idx % 16
            col_ld = (lane_idx / 16) * 8
            lds_ld_offset = (row_ld / 2) * Q_BYTES_PER_2ROWS + (row_ld % 2) * Q_ELEM_PER_ROW + col_ld
            lds_st_addr = p_lds_q_warp + lds_st_offset
            lds_st_ptr = _inttoptr_lds(lds_st_addr)
            lds_rd_addr = p_lds_q_warp + lds_ld_offset
            q_regs = []
            for p in range_constexpr(Q_NUM_PASSES):
                elem = base_elem + p * Q_ELEM_PER_ROW
                dword_off = _i32(ArithValue(elem) // 2)
                lo = buffer_ops.buffer_load(query_rsrc, dword_off, vec_width=4, dtype=T.i32)
                hi = buffer_ops.buffer_load(query_rsrc, _raw(ArithValue(dword_off) + 4), vec_width=4, dtype=T.i32)
                rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
                w0a, w1a = _bf16x4dw_to_fp8x2dw(lo)
                w0b, w1b = _bf16x4dw_to_fp8x2dw(hi)
                q_fp8 = Vec.from_elements([w0a, w1a, w0b, w1b], fx.Int32)
                rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                _ptr_store(q_fp8, lds_st_ptr, alignment=16)
                rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                q0 = _lds_load(lds_rd_addr, T.i64, static_byte_offset=0)
                q1 = _lds_load(lds_rd_addr, T.i64, static_byte_offset=MFMA_K)
                q_regs.append((q0, q1))
            q_nope_packs = []
            for p in range_constexpr(Q_NUM_PASSES):
                q_nope_packs.append(q_regs[p][0])
                q_nope_packs.append(q_regs[p][1])
            return q_nope_packs

        def _softmax_scale_p(idx_rsrc, num_rows_i32, p_vals, col_0_start, kv_end_i32):
            result = [None] * P_VALS_PER_THR
            for i in range_constexpr(P_VALS_PER_THR):
                result[i] = _f32(p_vals[i]) * qk_softmax_scale
            kv_end = _idx(kv_end_i32)
            skv = ArithValue(num_rows_i32)
            for i in range_constexpr(P_VALS_PER_THR):
                sub_offset = (i // 4) * 16 + (i % 4)
                pos = col_0_start + sub_offset
                slot = buffer_ops.buffer_load(idx_rsrc, _i32(pos), vec_width=1, dtype=T.i32)
                slot_a = ArithValue(slot)
                inv = ArithValue(_raw(pos >= kv_end))
                inv = inv | (slot_a < 0)
                inv = inv | (slot_a >= skv)
                result[i] = ArithValue(_raw(inv)).select(_raw(c_neg_inf), result[i])
            return result

        def _softmax(idx_rsrc, num_rows_i32, p_vals, row_max_old, row_sum_e_old, is_first, kv_tile_start_i32, kv_end_i32):
            col_0_start = lane_idx / 16 * 4 + _idx(kv_tile_start_i32)
            scaled = _softmax_scale_p(idx_rsrc, num_rows_i32, p_vals, col_0_start, kv_end_i32)
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

        def _process_tile_gemm1(idx_rsrc, num_rows_i32, p_lds_kv_base, kv_tile_start_i32, kv_end_i32, q_nope, rm_in, rse_in, is_first):
            k_base_i32 = _i32(ArithValue(p_lds_kv_base) + k_lds_lane_offset)
            P_COMP_SUBS = BLOCK_N // MFMA_N
            p_comp = [c_zero_v4f32] * P_COMP_SUBS
            for nope_pair in range_constexpr(NUM_NOPE_ITERS):
                tile_0 = nope_pair * 2
                tile_1 = nope_pair * 2 + 1
                k0 = [_load_k_from_lds(k_base_i32, 16 * h, tile_0 * BLOCK_K) for h in range_constexpr(P_COMP_SUBS)]
                k1 = [_load_k_from_lds(k_base_i32, 16 * h, tile_1 * BLOCK_K) for h in range_constexpr(P_COMP_SUBS)]
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
            p_vals = []
            for sub in range_constexpr(P_COMP_SUBS):
                pv = Vec(p_comp[sub])
                for ii in range_constexpr(4):
                    p_vals.append(pv[ii])
            v8_raw = _load_v_from_lds(p_lds_kv_base, warp_idx, lane_idx)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            rocdl.sched_barrier(0)
            p_exp_vals, rm_new, rse_new, rescale = _softmax(
                idx_rsrc, num_rows_i32, p_vals, rm_in, rse_in, is_first, kv_tile_start_i32, kv_end_i32
            )
            p_pack = _pack_p_to_fp8(p_exp_vals)
            vt8 = _transpose_v(v8_raw)
            _store_vt_to_lds(lds_base_idx + P_LDS_VT, warp_idx, lane_idx, vt8)
            return rm_new, rse_new, p_pack, rescale

        def _gemm2_core(p_pack, oaccu, vt_base_i32):
            for pv_pair in range_constexpr(NUM_PV_ITERS // 2):
                iter_a = pv_pair * 2
                iter_b = pv_pair * 2 + 1
                col_a_strip = iter_a * MFMA_N * 2
                col_b_strip = iter_b * MFMA_N * 2
                vta0_lo, vta0_hi = _load_vt_from_lds(vt_base_i32, col_a_strip)
                vta1_lo, vta1_hi = _load_vt_from_lds(vt_base_i32, col_a_strip + MFMA_N)
                vtb0_lo, vtb0_hi = _load_vt_from_lds(vt_base_i32, col_b_strip)
                vtb1_lo, vtb1_hi = _load_vt_from_lds(vt_base_i32, col_b_strip + MFMA_N)
                read0_lo = [vta0_lo, vtb0_lo]
                read0_hi = [vta0_hi, vtb0_hi]
                read1_lo = [vta1_lo, vtb1_lo]
                read1_hi = [vta1_hi, vtb1_hi]
                iter_idxs = [iter_a, iter_b]
                wait_lgkm = [4, 0]
                for step in range_constexpr(2):
                    rocdl.sched_barrier(0)
                    rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=wait_lgkm[step]))
                    lhs0 = _pack_i32x2(read0_lo[step], read0_hi[step])
                    lhs1 = _pack_i32x2(read1_lo[step], read1_hi[step])
                    iter_idx = iter_idxs[step]
                    acc_idx = iter_idx * 2
                    oaccu[acc_idx] = _mfma_fp8(T.f32x4, [lhs0, p_pack, oaccu[acc_idx], 0, 0, 0])
                    oaccu[acc_idx + 1] = _mfma_fp8(T.f32x4, [lhs1, p_pack, oaccu[acc_idx + 1], 0, 0, 0])
                rocdl.sched_barrier(0)
            return oaccu

        def _gemm2_first_iter(p_pack, vt_base_i32):
            _barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)
            oaccu = [c_zero_v4f32] * (NUM_PV_ITERS * 2)
            return _gemm2_core(p_pack, oaccu, vt_base_i32)

        def _gemm2_with_rescale(p_pack, rescale, oaccu_in, vt_base_i32):
            oaccu = _rescale_oaccu(oaccu_in, rescale)
            _barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)
            return _gemm2_core(p_pack, oaccu, vt_base_i32)

        def _pack_f32x4_to_bf16_2dw(acc_val):
            i16s = Vec(acc_val).to(fx.BFloat16).bitcast(fx.Int16)
            i16_0, i16_1, i16_2, i16_3 = (_raw(i16s[j]) for j in range(4))
            dw0 = _raw(ArithValue(i16_0).extui(T.i32) | (ArithValue(i16_1).extui(T.i32) << 16))
            dw1 = _raw(ArithValue(i16_2).extui(T.i32) | (ArithValue(i16_3).extui(T.i32) << 16))
            return dw0, dw1

        def _store_oaccu_pair_bf16(oaccu_a, oaccu_b, tile_idx, p_lds_o, row_base_i32):
            o16_row_st = lane_idx % 16
            o16_col_st = (lane_idx / 16) * 4
            o16_st_offset = _raw(
                ((o16_row_st / 2) * O16_ELEM_PER_PAD_2ROWS + (o16_row_st % 2) * O16_NUM_COLS + o16_col_st) * 2
            )
            o16_row_ld = lane_idx / 4
            o16_col_ld = (lane_idx % 4) * 8
            o16_rd_offset = _raw(
                ((o16_row_ld / 2) * O16_ELEM_PER_PAD_2ROWS + (o16_row_ld % 2) * O16_NUM_COLS + o16_col_ld) * 2
            )
            lds_warp = ArithValue(p_lds_o) + warp_idx * O16_LDS_PER_WARP
            lds_st_addr = _i32(ArithValue(lds_warp) + o16_st_offset)
            for sub, acc_val in enumerate([oaccu_a, oaccu_b]):
                dw0, dw1 = _pack_f32x4_to_bf16_2dw(acc_val)
                vec_2dw = Vec.from_elements([dw0, dw1], fx.Int32)
                st_addr_sub = _i32(ArithValue(lds_st_addr) + sub * O16_NUM_COLS)
                _ptr_store(vec_2dw, _lds_ptr_from_i32(st_addr_sub), alignment=8, volatile_=True)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            lds_rd_addr = _i32(ArithValue(lds_warp) + o16_rd_offset)
            data = _ptr_load(T.i32x4, _lds_ptr_from_i32(lds_rd_addr), alignment=16)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            row_vram = ArithValue(row_base_i32) + o16_row_ld
            col_vram = ArithValue(o16_col_ld) + tile_idx * MFMA_N * 2
            vram_offset = _raw((row_vram * V_HEAD_DIM + col_vram) * 2)
            buffer_ops.buffer_store(data, final_output_rsrc, vram_offset, offset_is_bytes=True)

        def _normalize_and_store(oaccu, rm, rse, row_base_idx):
            p_lds_o = p_lds_kv_0_base
            # sink: fold a per-head virtual key (score=sink[h], zero value).
            if const_expr(HAS_SINK):
                head = _i32(ArithValue(_uniform_i32(warp_idx)) * 16 + ArithValue(_i32(lane_idx % 16)))
                sink_val = _f32(buffer_ops.buffer_load(sink_rsrc, head, vec_width=1, dtype=T.f32))
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
            if const_expr(USE_PT_SCALE):
                scl = _fmul(scl, kv_sc)
            scl_vec = _raw(Vec.filled(4, _f32(scl), fx.Float32))
            _barrier(lgkmcnt=0)
            for pv in range_constexpr(NUM_PV_ITERS):
                a0 = _f32(oaccu[pv * 2]) * scl_vec
                a1 = _f32(oaccu[pv * 2 + 1]) * scl_vec
                a0 = ArithValue(_raw(valid_i1)).select(_raw(a0), _raw(c_zero_v4f32))
                a1 = ArithValue(_raw(valid_i1)).select(_raw(a1), _raw(c_zero_v4f32))
                _store_oaccu_pair_bf16(a0, a1, pv, p_lds_o, row_base_idx)

        p_lds_kv_0_base = lds_base_idx + P_LDS_KV_0

        def _kv_warp_lds_base(p_lds_kv_base):
            warp_offset = _raw(ArithValue(_uniform_i32(warp_idx)) * KV_SUB_BYTES)
            return _raw(ArithValue(_i32(p_lds_kv_base)) + warp_offset)

        p_lds_kv_0_warp = _kv_warp_lds_base(p_lds_kv_0_base)

        # ---- CSR ranges ----
        main_rng = Vec(buffer_ops.buffer_load(main_indptr_rsrc, q_idx, vec_width=2, dtype=T.i32))
        main_start = rocdl.readfirstlane(T.i32, main_rng[0])
        main_end = rocdl.readfirstlane(T.i32, main_rng[1])
        main_len = _raw(ArithValue(main_end) - ArithValue(main_start))
        n0_tiles = _raw((ArithValue(main_len) + (BLOCK_N - 1)).with_signedness(False) // BLOCK_N)

        if const_expr(NREG == 2):
            extra_rng = Vec(buffer_ops.buffer_load(extra_indptr_rsrc, q_idx, vec_width=2, dtype=T.i32))
            extra_start = rocdl.readfirstlane(T.i32, extra_rng[0])
            extra_end = rocdl.readfirstlane(T.i32, extra_rng[1])
            extra_len = _raw(ArithValue(extra_end) - ArithValue(extra_start))
            n1_tiles = _raw((ArithValue(extra_len) + (BLOCK_N - 1)).with_signedness(False) // BLOCK_N)
            total_tiles = _raw(ArithValue(n0_tiles) + ArithValue(n1_tiles))
        else:
            total_tiles = n0_tiles

        row_base = _idx(q_idx) * NUM_QO_HEADS + warp_idx * 16
        q_nope_packs = _load_q_to_regs(q_idx)

        # ---- region-0 attend body (B1 single-region: const-fixed resources) ----
        def _attend_region0(kv_tile_start_i32, rm_in, rse_in, oaccu_in, is_first):
            tb, sb = _row_addrs(
                main_indices_rsrc, main_bt_rsrc, main_num_rows, main_block_size, main_max_blocks,
                kv_tile_start_i32, main_end,
            )
            if const_expr(GLM_FLAT):
                _load_flat_dma(main_cache_rsrc, p_lds_kv_0_warp, tb)
            else:
                if const_expr(R0_CONVERT):
                    _load_nope_convert(
                        main_cache_rsrc, p_lds_kv_0_warp, tb, sb,
                        fx.Float32(1.0) if R0_OCP else fx.Float32(0.0),
                    )
                else:
                    _load_nope_dma(main_cache_rsrc, p_lds_kv_0_warp, tb)
                _load_rope_block(main_cache_rsrc, p_lds_kv_0_warp, tb)
            _barrier(vmcnt=0, lgkmcnt=0)
            rocdl.sched_barrier(0)
            rm_n, rse_n, p_pack, rescale = _process_tile_gemm1(
                main_indices_rsrc, main_num_rows, p_lds_kv_0_base, kv_tile_start_i32, main_end,
                q_nope_packs, rm_in, rse_in, is_first,
            )
            if const_expr(is_first):
                oaccu_n = _gemm2_first_iter(p_pack, _vt_base_i32())
            else:
                oaccu_n = _gemm2_with_rescale(p_pack, rescale, oaccu_in, _vt_base_i32())
            return rm_n, rse_n, oaccu_n

        # ---- region-select load + GEMM1 (B2) ----------------------------------
        # arith.select on the !llvm.ptr<8> buffer descriptors does NOT lower
        # correctly, and two sequential yield-loops in one body break the
        # structured-for lowering. So B2 uses ONE yield-loop with a runtime
        # ``if`` that selects the region for the load + GEMM1 (each region keeps
        # compile-time-fixed resources). Only flat scalars cross the if
        # (rm, rse, the i64 P-pack, rescale) -- the GEMM2 P@V accumulation reads
        # V^T from LDS and is region-agnostic, so it runs after the if. This is
        # the same scalar-carry-across-runtime-if pattern as the MLA decode
        # kernel. Tile space: region0 tiles [0, n0) then region1 tiles [0, n1),
        # flattened to global tile g with is_r1 = g >= n0_tiles.
        def _load_gemm1_select(global_t_i32, rm_in, rse_in, is_first):
            is_r1 = _raw(ArithValue(global_t_i32) >= ArithValue(n0_tiles))
            rm_n = c_neg_large
            rse_n = c_zero_f32
            pp = fx.Int64(0)
            rescale = c_one_f32
            if is_r1:
                local = _raw(ArithValue(global_t_i32) - ArithValue(n0_tiles))
                kv_ts = _raw(ArithValue(extra_start) + ArithValue(local) * BLOCK_N)
                tb, sb = _row_addrs(
                    extra_indices_rsrc, extra_bt_rsrc, extra_num_rows, extra_block_size,
                    extra_max_blocks, kv_ts, extra_end,
                )
                _load_nope_convert(
                    extra_cache_rsrc, p_lds_kv_0_warp, tb, sb,
                    fx.Float32(1.0) if R1_OCP else fx.Float32(0.0),
                )
                _load_rope_block(extra_cache_rsrc, p_lds_kv_0_warp, tb)
                _barrier(vmcnt=0, lgkmcnt=0)
                rocdl.sched_barrier(0)
                rm_n, rse_n, pp, rescale = _process_tile_gemm1(
                    extra_indices_rsrc, extra_num_rows, p_lds_kv_0_base, kv_ts, extra_end,
                    q_nope_packs, rm_in, rse_in, is_first,
                )
            else:
                kv_ts = _raw(ArithValue(main_start) + ArithValue(global_t_i32) * BLOCK_N)
                tb, sb = _row_addrs(
                    main_indices_rsrc, main_bt_rsrc, main_num_rows, main_block_size,
                    main_max_blocks, kv_ts, main_end,
                )
                if const_expr(R0_CONVERT):
                    _load_nope_convert(
                        main_cache_rsrc, p_lds_kv_0_warp, tb, sb,
                        fx.Float32(1.0) if R0_OCP else fx.Float32(0.0),
                    )
                else:
                    _load_nope_dma(main_cache_rsrc, p_lds_kv_0_warp, tb)
                _load_rope_block(main_cache_rsrc, p_lds_kv_0_warp, tb)
                _barrier(vmcnt=0, lgkmcnt=0)
                rocdl.sched_barrier(0)
                rm_n, rse_n, pp, rescale = _process_tile_gemm1(
                    main_indices_rsrc, main_num_rows, p_lds_kv_0_base, kv_ts, main_end,
                    q_nope_packs, rm_in, rse_in, is_first,
                )
            return rm_n, rse_n, pp, rescale

        if const_expr(NREG == 1):
            # First tile is always region0 tile 0 (handles all-empty -> masked
            # -> zero output, and initialises the shared softmax state).
            rm_first, rse_first, oaccu_first = _attend_region0(
                main_start, c_neg_large, c_zero_f32, None, True
            )
            has_multi = ArithValue(n0_tiles) > 1

            def _multi_tile_path():
                init_args = [rm_first, rse_first] + oaccu_first
                for tile_iv, state in range(_idx(1), _idx(n0_tiles), _idx(1), init=init_args):
                    tile_i32 = _raw(ArithValue(fx.Int32(tile_iv)))
                    kv_ts = _raw(ArithValue(main_start) + ArithValue(tile_i32) * BLOCK_N)
                    rm_c = state[0]
                    rse_c = state[1]
                    oaccu_c = [state[2 + i] for i in range(NUM_PV_ITERS * 2)]
                    rm_n, rse_n, oaccu_n = _attend_region0(kv_ts, rm_c, rse_c, oaccu_c, False)
                    results = yield [rm_n, rse_n] + oaccu_n
                rm_final = results[0]
                rse_final = results[1]
                oaccu_final = [results[2 + i] for i in range(NUM_PV_ITERS * 2)]
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
        else:
            # B2: single yield-loop over the flattened region0||region1 tile
            # space. The first tile (g==0) is region0 tile 0 (region0/SWA is the
            # always-present sliding window) and initialises the shared softmax
            # state via GEMM2-first; the loop selects the region per tile.
            rm_first, rse_first, oaccu_first = _attend_region0(
                main_start, c_neg_large, c_zero_f32, None, True
            )
            has_multi = ArithValue(total_tiles) > 1

            def _multi_tile_path2():
                init_args = [rm_first, rse_first] + oaccu_first
                for tile_iv, state in range(_idx(1), _idx(total_tiles), _idx(1), init=init_args):
                    tile_i32 = _raw(ArithValue(fx.Int32(tile_iv)))
                    rm_c = state[0]
                    rse_c = state[1]
                    oaccu_c = [state[2 + i] for i in range(NUM_PV_ITERS * 2)]
                    rm_n, rse_n, pp, rescale = _load_gemm1_select(tile_i32, rm_c, rse_c, False)
                    oaccu_n = _gemm2_with_rescale(pp, rescale, oaccu_c, _vt_base_i32())
                    results = yield [rm_n, rse_n] + oaccu_n
                rm_final = results[0]
                rse_final = results[1]
                oaccu_final = [results[2 + i] for i in range(NUM_PV_ITERS * 2)]
                _normalize_and_store(oaccu_final, rm_final, rse_final, row_base)

            def _single_tile_path2():
                _normalize_and_store(oaccu_first, rm_first, rse_first, row_base)

            @flyc.jit
            def _dispatch2():
                if has_multi:
                    _multi_tile_path2()
                else:
                    _single_tile_path2()

            _dispatch2()

    @flyc.jit
    def launch_paged(
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
        q_scale: fx.Tensor,
        kv_scale: fx.Tensor,
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
        kn_sparse_mla_prefill_paged(
            query, main_cache, main_indices, main_indptr, main_block_table,
            extra_cache, extra_indices, extra_indptr, extra_block_table,
            q_req, sink_buf, final_output, q_scale, kv_scale, SOFTMAX_SCALE,
            main_num_rows, extra_num_rows, main_block_size, extra_block_size,
            main_max_blocks, extra_max_blocks,
        ).launch(grid=(grid_x, 1, 1), block=(NUM_THREADS, 1, 1), smem=0, stream=stream)

    return launch_paged


# ---------------------------------------------------------------------------
# Builder (constexpr specialization entry point)
# ---------------------------------------------------------------------------
def compile_sparse_mla_prefill(
    *,
    num_q_heads: int = 128,
    head_dim: int = 512,
    v_dim: int = 512,
    num_regions: int = 1,
    has_sink: bool = False,
    region0_dtype: str = "fp8",
    region0_is_fnuz: bool = True,
    region1_dtype: str = "fp8",
    region1_is_fnuz: bool = True,
    qk_split: bool = False,
    block_n: int = 32,
    block_h: int = 16,
    split_kv: bool = False,
    waves_per_eu: int = 2,
    packed: bool = False,
    scale_mode: str = "none",
    softmax_scale: float | None = None,
    single_request: bool = True,
    cache_layout: str = "fp8_ds_mla",
):
    """Return the compiled launcher for the sparse MLA prefill kernel.

    ``packed=False`` (default) returns the Phase A flat-cache launcher (512
    only). With ``packed=True`` it returns the paged launcher:

      - DSv4 (``head_dim=512``, ``cache_layout='fp8_ds_mla'``): single/two
        region, optional sink, UE8M0/OCP via the convert load path.
      - GLM/DSv3.2 (``head_dim=576``, ``cache_layout='glm_flat576'``):
        single-region flat fp8 cache, ``scale_mode='per_tensor'`` (runtime
        ``q_scale`` / ``kv_scale`` f32 [1] launch args), no sink.
    """
    if num_q_heads != NUM_QO_HEADS:
        raise NotImplementedError(f"requires num_q_heads={NUM_QO_HEADS}, got {num_q_heads}")
    if v_dim != V_HEAD_DIM:
        raise NotImplementedError(f"requires v_dim={V_HEAD_DIM}, got {v_dim}")
    if head_dim not in (512, 576):
        raise NotImplementedError(f"requires head_dim in (512, 576), got {head_dim}")
    if split_kv:
        raise NotImplementedError("split_kv is Phase C")

    if not packed:
        if block_n != BLOCK_N:
            raise NotImplementedError(f"flat (Phase A) path supports block_n={BLOCK_N} only")
        if head_dim != QK_HEAD_DIM:
            raise NotImplementedError("flat (Phase A) path supports head_dim=512 only; use packed=True")
        if num_regions != 1 or has_sink or qk_split:
            raise NotImplementedError("Phase A flat path: num_regions=1, has_sink=False, qk_split=False")
        if region0_dtype != "fp8":
            raise NotImplementedError("Phase A: region0_dtype must be 'fp8' (native fp8 MFMA)")
        sm = fx.Float32(QK_HEAD_DIM ** -0.5 if softmax_scale is None else float(softmax_scale))

        @flyc.jit
        def launch_flat(
            query: fx.Tensor,
            kv_buffer: fx.Tensor,
            indices: fx.Tensor,
            indptr: fx.Tensor,
            final_output: fx.Tensor,
            num_queries: fx.Int32,
            num_kv_rows: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_sparse_mla_prefill(
                query, kv_buffer, indices, indptr, final_output, sm, num_queries, num_kv_rows, stream
            )

        return launch_flat

    if num_regions not in (1, 2):
        raise NotImplementedError("packed path supports num_regions in {1, 2}")
    if cache_layout == "glm_flat576":
        if head_dim != 576 or num_regions != 1 or has_sink:
            raise NotImplementedError("glm_flat576: head_dim=576, single-region, no sink")
        if scale_mode != "per_tensor":
            raise NotImplementedError("glm_flat576 requires scale_mode='per_tensor'")
    elif head_dim != 512:
        raise NotImplementedError("fp8_ds_mla cache_layout requires head_dim=512")
    # region0 (SWA) is fnuz; UE8M0 != 1 forces the convert load path. GLM flat
    # rows are read straight to LDS (no per-block convert).
    r0_convert = (scale_mode == "ue8m0") or (not region0_is_fnuz)
    return compile_sparse_mla_prefill_paged(
        num_regions=num_regions,
        has_sink=has_sink,
        r0_convert=r0_convert,
        r0_is_ocp=(not region0_is_fnuz),
        r1_is_ocp=(not region1_is_fnuz),
        waves_per_eu=waves_per_eu,
        softmax_scale=softmax_scale,
        single_request=single_request,
        head_dim=head_dim,
        v_dim=v_dim,
        cache_layout=cache_layout,
        scale_mode=scale_mode,
        block_n=block_n,
    )
