# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MLA decode kernel (nhead=128, fp8 Q, fp8 KV, bf16 output).

Transplanted from csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h128_fp8_fp8.cuh.

Architecture: 8 warps / 512 threads, persistent-thread dispatch.
Per work item: load Q -> iterate KV tiles (BLOCK_N=32) -> QK GEMM (nope+rope)
-> online softmax -> PV GEMM -> output (final bf16 or split f32 + LSE).

NOTE: Do NOT use ``from __future__ import annotations`` here -- it breaks
``fx.Constexpr`` detection in the FlyDSL AST rewriter.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl._mlir.dialects import arith as _std_arith
from flydsl._mlir.dialects import math as _math
from flydsl._mlir.dialects import memref as _memref
from flydsl._mlir.dialects import gpu as _mlir_gpu
from flydsl._mlir.dialects._arith_enum_gen import CmpIPredicate

from flydsl.expr import arith, vector, gpu, buffer_ops, rocdl
from flydsl.expr import range_constexpr
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T

from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.runtime.device import get_rocm_arch as get_hip_arch


# ---------------------------------------------------------------------------
# Compile-time constants (mirroring HkMlaDecodeFwdTraits)
# ---------------------------------------------------------------------------
NUM_QO_HEADS: int = 128
NUM_KV_HEADS: int = 1
KV_LORA_RANK: int = 512
QK_NOPE_HEAD_DIM: int = KV_LORA_RANK  # 512
QK_ROPE_HEAD_DIM: int = 64
QK_HEAD_DIM: int = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM: int = KV_LORA_RANK  # 512
PAGE_SIZE: int = 1
NUM_WARPS: int = 8
WARP_SIZE: int = 64
NUM_THREADS: int = NUM_WARPS * WARP_SIZE  # 512
BLOCK_M: int = 128  # == NUM_QO_HEADS
BLOCK_N: int = 32
BLOCK_K: int = 32
TILE_M: int = BLOCK_M // NUM_WARPS  # 16
OCCUPANCY: int = 1

SIZE_MLA_WORK_INFO_IN_DW: int = 8
LOG2E: float = 1.4426950408889634

# ---------------------------------------------------------------------------
# KvManagerV2 LDS layout constants
# ---------------------------------------------------------------------------
# KV tile: 32 rows x 576 cols (fp8), split into 9 blocks of 64 cols each.
# Each block: 8 sub-blocks (one per warp) of 4 rows x 64 cols + 2 DW padding.
KV_NUM_COLS: int = 64
KV_NUM_BLOCKS: int = QK_HEAD_DIM // KV_NUM_COLS  # 576 / 64 = 9
KV_ROWS_PER_SUB: int = BLOCK_N // NUM_WARPS  # 32 / 8 = 4
KV_BYTES_PER_ROW: int = KV_NUM_COLS  # 64 * 1 (fp8)
KV_PAD_DW: int = 2
KV_SUB_BYTES: int = KV_ROWS_PER_SUB * KV_BYTES_PER_ROW + KV_PAD_DW * 4  # 264
KV_NUM_SUBS: int = BLOCK_N // KV_ROWS_PER_SUB  # 8
KV_BLOCK_BYTES: int = KV_SUB_BYTES * KV_NUM_SUBS  # 2112
SZ_LDS_KV: int = KV_BLOCK_BYTES * KV_NUM_BLOCKS  # 2112 * 9 = 19008

# ---------------------------------------------------------------------------
# VtManagerV1 LDS layout constants
# ---------------------------------------------------------------------------
VT_ROWS_PER_THR: int = 4
VT_COLS_PER_THR: int = 8
VT_ELEMS_PER_BLK: int = VT_ROWS_PER_THR * VT_COLS_PER_THR  # 32
VT_BLKS_PER_ROW: int = V_HEAD_DIM // VT_COLS_PER_THR  # 64
VT_BLKS_PER_ROW_PAD: int = VT_BLKS_PER_ROW + 2  # 66
VT_NUM_SUB_BLKS: int = 8
SZ_LDS_VT: int = VT_NUM_SUB_BLKS * (
    (BLOCK_N // VT_NUM_SUB_BLKS) * V_HEAD_DIM + 16 * 4
)  # 8 * (4*512 + 64) = 16896

# ---------------------------------------------------------------------------
# QManagerV3 LDS layout constants (per-warp staging for VRAM->LDS->GPR)
# ---------------------------------------------------------------------------
Q_ELEM_PER_ROW: int = 64
Q_ELEM_PER_COL: int = 16
Q_PAD_BYTES_PER_2ROWS: int = 8  # 2 DW
Q_BYTES_PER_2ROWS: int = Q_ELEM_PER_ROW * 2 + Q_PAD_BYTES_PER_2ROWS  # 136
SZ_LDS_Q_PER_WARP: int = Q_ELEM_PER_COL // 2 * Q_BYTES_PER_2ROWS  # 1088
SZ_LDS_Q: int = NUM_WARPS * SZ_LDS_Q_PER_WARP  # 8704

# ---------------------------------------------------------------------------
# OManager16bitsV2 (bf16 output via LDS reshape)
# ---------------------------------------------------------------------------
O16_NUM_ROWS: int = 16
O16_NUM_COLS: int = 32
O16_PAD_ELEM_PER_2ROWS: int = 4  # padded 2-row stride in bf16 elements
O16_ELEM_PER_PAD_2ROWS: int = 2 * O16_NUM_COLS + O16_PAD_ELEM_PER_2ROWS  # 68
O16_LDS_PER_WARP: int = (O16_NUM_ROWS // 2) * O16_ELEM_PER_PAD_2ROWS * 2  # 1088
SZ_LDS_O16: int = NUM_WARPS * O16_LDS_PER_WARP  # 8704  (reuses p_lds_kv region)

# ---------------------------------------------------------------------------
# OManager32bitsV2 (f32 split output via LDS reshape)
# ---------------------------------------------------------------------------
O32_NUM_ROWS: int = 16
O32_NUM_COLS: int = 32
O32_PAD_ELEM_PER_ROW: int = 4
O32_ELEM_PER_PAD_ROW: int = O32_NUM_COLS + O32_PAD_ELEM_PER_ROW  # 36
O32_LDS_PER_WARP: int = O32_NUM_ROWS * O32_ELEM_PER_PAD_ROW * 4  # 2304
SZ_LDS_O32: int = NUM_WARPS * O32_LDS_PER_WARP  # 18432

# Overall LDS layout (byte offsets):
#   [0, SZ_LDS_VT) = Vt staging buffer
#   [SZ_LDS_VT, SZ_LDS_VT + SZ_LDS_Q) = Q staging buffer
#   [SZ_LDS_VT + SZ_LDS_Q, +SZ_LDS_KV) = KV double-buffer 0
#   [SZ_LDS_VT + SZ_LDS_Q + SZ_LDS_KV, +SZ_LDS_KV) = KV double-buffer 1
# Output reuses the KV double-buffer 0 region.
P_LDS_VT: int = 0
P_LDS_Q: int = SZ_LDS_VT  # 16896
P_LDS_KV_0: int = P_LDS_Q + SZ_LDS_Q  # 25600
P_LDS_KV_1: int = P_LDS_KV_0 + SZ_LDS_KV  # 44608
TOTAL_LDS_BYTES: int = P_LDS_KV_1 + SZ_LDS_KV  # 63616

assert (
    max(SZ_LDS_O16, SZ_LDS_O32) <= SZ_LDS_KV
), "Output LDS must fit in one KV buffer region"

# ---------------------------------------------------------------------------
# MFMA tile constants
# ---------------------------------------------------------------------------
MFMA_M: int = 16
MFMA_N: int = 16
MFMA_K: int = 32  # mfma_f32_16x16x32_fp8_fp8
MFMA_ELEM_PER_THR: int = MFMA_M * MFMA_K // WARP_SIZE  # 8

# Number of QK sub-tile iterations
NUM_NOPE_ITERS: int = QK_NOPE_HEAD_DIM // (MFMA_K * 2)  # 512/64 = 8
NUM_ROPE_ITERS: int = QK_ROPE_HEAD_DIM // (MFMA_K * 2)  # 64/64 = 1
NUM_PV_ITERS: int = V_HEAD_DIM // (MFMA_N * 2)  # 512/32 = 16


# ---------------------------------------------------------------------------
# Utility helpers (ported from FlyDSL/kernels/mla_decode_fp8.py)
# ---------------------------------------------------------------------------


def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


def _barrier(vmcnt=63, lgkmcnt=63):
    """Emit s_waitcnt + s_barrier via inline asm."""
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
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="\n".join(parts),
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


_LDS_PTR_TYPE = None


def _inttoptr_lds(i64_val):
    """Convert i64 scalar to !llvm.ptr<3> (LDS pointer)."""
    global _LDS_PTR_TYPE
    if _LDS_PTR_TYPE is None:
        _LDS_PTR_TYPE = ir.Type.parse("!llvm.ptr<3>")
    return llvm.inttoptr(_LDS_PTR_TYPE, i64_val)


def _get_element_ptr(base_ptr, byte_offset=None, static_byte_offset=0, elem_type=None):
    """GEP-based pointer arithmetic."""
    _GEP_DYN = -(2**31)
    raw_ptr = _raw(base_ptr) if not isinstance(base_ptr, ir.Value) else base_ptr
    if elem_type is None:
        elem_type = T.i8

    if byte_offset is None:
        return llvm.GEPOp(
            raw_ptr.type,
            raw_ptr,
            [],
            [int(static_byte_offset)],
            elem_type,
            None,
        ).result
    elif isinstance(byte_offset, int):
        return llvm.GEPOp(
            raw_ptr.type,
            raw_ptr,
            [],
            [int(byte_offset) + int(static_byte_offset)],
            elem_type,
            None,
        ).result
    else:
        offset_val = (
            _raw(byte_offset) if not isinstance(byte_offset, ir.Value) else byte_offset
        )
        if isinstance(offset_val.type, ir.IndexType):
            offset_val = _std_arith.IndexCastOp(T.i64, offset_val).result
        if static_byte_offset != 0:
            static_attr = ir.IntegerAttr.get(offset_val.type, int(static_byte_offset))
            static_const = _std_arith.ConstantOp(offset_val.type, static_attr).result
            offset_val = _std_arith.AddIOp(offset_val, static_const).result
        return llvm.GEPOp(
            raw_ptr.type,
            raw_ptr,
            [offset_val],
            [_GEP_DYN],
            elem_type,
            None,
        ).result


def _lds_load(byte_addr_index, vec_type, static_byte_offset=0):
    """LDS load via raw llvm.LoadOp on an LDS pointer (addr space 3)."""
    raw_addr = (
        _raw(byte_addr_index)
        if not isinstance(byte_addr_index, ir.Value)
        else byte_addr_index
    )
    addr_i64 = _std_arith.IndexCastOp(T.i64, raw_addr).result
    lds_ptr = _inttoptr_lds(addr_i64)
    if static_byte_offset != 0:
        lds_ptr = _get_element_ptr(lds_ptr, static_byte_offset=static_byte_offset)
    return llvm.LoadOp(vec_type, lds_ptr, alignment=16, nontemporal=True).result


def _index_cast_to_i32(value):
    """Cast index/ArithValue to i32.  No-op if already i32."""
    raw = _raw(value) if not isinstance(value, ir.Value) else value
    if raw.type == T.i32:
        return raw
    return _std_arith.IndexCastOp(T.i32, raw).result


def _fast_exp2(val):
    """Bare v_exp_f32 via rocdl.exp2 -- no range reduction."""
    return rocdl.exp2(T.f32, _raw(val))


def _to_mlir(val, index=False):
    """Convert Python int/float, ArithValue, or ir.Value to raw MLIR Value."""
    if isinstance(val, int):
        return _raw(arith.constant(val, index=index))
    if isinstance(val, float):
        return _raw(arith.constant(val))
    if isinstance(val, ir.Value):
        return val
    return _raw(val)


def _set_mfma_vgpr_form():
    """Force MFMA to use ACC_CD=0 (D/C in ArchVGPR) via LLVM cl::opt."""
    import ctypes
    import os

    lib_dir = os.path.dirname(
        __import__("flydsl._mlir._mlir_libs", fromlist=["_mlir_libs"]).__file__
    )
    lib_name = "libFlyPythonCAPI.so"
    lib_path = os.path.join(lib_dir, lib_name)
    if not os.path.exists(lib_path):
        lib_name = "libFlirPythonCAPI.so"
        lib_path = os.path.join(lib_dir, lib_name)
    lib = ctypes.CDLL(lib_path)
    try:
        parse_fn = lib.LLVMParseCommandLineOptions
    except AttributeError:
        return  # Symbol not available in this build
    parse_fn.restype = None
    parse_fn.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_char_p,
    ]
    argv = [
        b"mlir",
        b"-amdgpu-mfma-vgpr-form",
        b"--amdgpu-schedule-metric-bias=0",
        b"--enable-deferred-spilling",
    ]
    argv_arr = (ctypes.c_char_p * len(argv))(*argv)
    parse_fn(len(argv), argv_arr, None)


_set_mfma_vgpr_form()


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
def kn_mla_fwd_decode_h128_fp8_fp8(
    # --- inputs ---
    query: fx.Tensor,  # [num_seqs * num_heads, qk_head_dim]  (fp8)
    kv_buffer: fx.Tensor,  # [num_pages, qk_head_dim]  (fp8)
    kv_page_indices: fx.Tensor,  # [num_page_used]  (i32)
    # --- metadata ---
    work_indptr: fx.Tensor,  # [num_workers + 1]  (i32)
    work_info_set: fx.Tensor,  # [num_work_items * 8]  (i32)
    # --- outputs ---
    final_output: fx.Tensor,  # [num_seqs * num_heads, v_head_dim]  (bf16)
    split_output: fx.Tensor,  # [num_partial_slots * num_heads, v_head_dim]  (f32)
    split_lse: fx.Tensor,  # [num_partial_slots * num_heads]  (f32)
    # --- parameters ---
    softmax_scale: fx.Float32,
):
    """MLA decode forward kernel (nhead=128, fp8/fp8 -> bf16).

    Persistent-thread kernel: each workgroup picks up work items
    from ``work_indptr`` / ``work_info_set`` and processes them sequentially.
    """
    _STUB_EARLY_RETURN = False  # Set True to skip all kernel body for testing launch
    if _STUB_EARLY_RETURN:
        return

    # ---- Types ----
    fm_fast = _std_arith.FastMathFlags.fast
    # fastmath without ninf: safe for operations that may encounter -inf
    # (boundary masking sets OOB attention scores to -inf)
    fm_no_inf = (
        _std_arith.FastMathFlags.nnan
        | _std_arith.FastMathFlags.nsz
        | _std_arith.FastMathFlags.arcp
        | _std_arith.FastMathFlags.contract
        | _std_arith.FastMathFlags.afn
        | _std_arith.FastMathFlags.reassoc
    )

    def _mfma_fp8(result_type, operands, **kw):
        return rocdl.mfma_f32_16x16x32_fp8_fp8(result_type, operands, **kw)

    # ---- LDS setup ----
    arch = get_hip_arch()
    lds_allocator = SmemAllocator(None, arch=arch)
    lds_allocator.ptr = TOTAL_LDS_BYTES  # reserve LDS bytes

    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        lds_allocator.finalize()

    lds_buffer = lds_allocator.get_base()
    lds_base_idx = _memref.ExtractAlignedPointerAsIndexOp(lds_buffer).result

    # ---- V^T transpose perm constants ----
    c_perm0 = arith.constant(0x05010400, type=T.i32)
    c_perm1 = arith.constant(0x07030602, type=T.i32)
    c_perm2 = arith.constant(0x05040100, type=T.i32)
    c_perm3 = arith.constant(0x07060302, type=T.i32)

    def _vt_perm(src_hi, src_lo, sel):
        return llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.perm",
            [src_hi, src_lo, sel],
            [],
            [],
        )

    # ---- Constants ----
    c_neg_inf_f32 = arith.constant(float("-inf"), type=T.f32)
    c_zero_f32 = arith.constant(0.0, type=T.f32)
    c_one_f32 = arith.constant(1.0, type=T.f32)
    c_zero_i32 = arith.constant(0, type=T.i32)
    c_zero_v4f32 = arith.constant_vector(0.0, T.f32x4)
    c_log2e = arith.constant(LOG2E, type=T.f32)
    c_inv_log2e = arith.constant(1.0 / LOG2E, type=T.f32)
    c_dword_sz = arith.constant(4, type=T.i32)
    c_aux_zero = arith.constant(0, type=T.i32)

    # ---- Buffer resources ----
    query_rsrc = buffer_ops.create_buffer_resource(query)
    kv_rsrc = buffer_ops.create_buffer_resource(kv_buffer)
    kv_page_indices_rsrc = buffer_ops.create_buffer_resource(kv_page_indices)
    work_indptr_rsrc = buffer_ops.create_buffer_resource(work_indptr)
    work_info_set_rsrc = buffer_ops.create_buffer_resource(work_info_set)
    final_output_rsrc = buffer_ops.create_buffer_resource(final_output)
    split_output_rsrc = buffer_ops.create_buffer_resource(split_output)
    split_lse_rsrc = buffer_ops.create_buffer_resource(split_lse)

    # ---- Thread indices ----
    worker_idx = gpu.block_idx.x
    tid = gpu.thread_id("x")
    warp_idx = tid / arith.index(WARP_SIZE)
    lane_idx = tid % arith.index(WARP_SIZE)
    warp_idx_i32 = rocdl.readfirstlane(T.i32, _raw(_index_cast_to_i32(warp_idx)))
    lane_idx_i32 = _index_cast_to_i32(lane_idx)

    # ---- Work range ----
    worker_idx_i32 = _index_cast_to_i32(worker_idx)
    work_range = buffer_ops.buffer_load(
        work_indptr_rsrc, worker_idx_i32, vec_width=2, dtype=T.i32
    )
    work_start_i32 = rocdl.readfirstlane(T.i32, _raw(vector.extract(work_range, [0])))
    work_end_i32 = rocdl.readfirstlane(T.i32, _raw(vector.extract(work_range, [1])))
    work_start_idx = arith.index_cast(T.index, work_start_i32)
    work_end_idx = arith.index_cast(T.index, work_end_i32)

    # ---- KvManagerV2 thread-to-data mapping ----
    # Each warp takes 4 rows: warp w -> rows {w*2, w*2+1, w*2+16, w*2+17}
    # lane mapping: (lane/32)*16 + (lane/16)%2 + warp*2
    kv_ld_row_base = (
        lane_idx / arith.index(32) * arith.index(16)
        + (lane_idx / arith.index(16)) % arith.index(2)
        + warp_idx * arith.index(2)
    )
    kv_ld_row_base_i32 = _index_cast_to_i32(kv_ld_row_base)
    kv_ld_col_base_i32 = _index_cast_to_i32(
        (lane_idx % arith.index(16)) * arith.index(4)
    )

    # ---- Helper: resolve KV page index -> physical row ----
    def _get_kv_ld_row(kv_tile_start_i32, kv_tile_end_i32, check_boundary):
        """Resolve physical KV row for this thread's assigned row.

        For OOB rows (row >= kv_end), returns -1 WITHOUT issuing a
        buffer_load -- avoids reading garbage from kv_page_indices.
        """
        row_idx_i32 = _std_arith.AddIOp(
            _raw(kv_ld_row_base_i32), _raw(kv_tile_start_i32)
        ).result
        if check_boundary:
            neg_one = _raw(arith.constant(-1, type=T.i32))
            if_op = scf.IfOp(
                _std_arith.CmpIOp(
                    CmpIPredicate.slt, row_idx_i32, _raw(kv_tile_end_i32)
                ).result,
                [T.i32],
                has_else=True,
            )
            with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                # In-bounds: do the buffer_load
                phys_row_ib = buffer_ops.buffer_load(
                    kv_page_indices_rsrc, row_idx_i32, vec_width=1, dtype=T.i32
                )
                scf.YieldOp([_raw(phys_row_ib)])
            with ir.InsertionPoint(if_op.regions[1].blocks[0]):
                # OOB: return -1
                scf.YieldOp([neg_one])
            return if_op.results[0]
        else:
            phys_row = buffer_ops.buffer_load(
                kv_page_indices_rsrc, row_idx_i32, vec_width=1, dtype=T.i32
            )
            return _raw(phys_row)

    # ---- Helper: async_load_k_tile (VRAM->LDS via buffer_load_dword_lds) ----
    def _async_load_k_tile(
        p_lds_kv_warp_i32, row_i32, col_base_i32, block_idx_const, check_boundary=False
    ):
        """Load one 32x64 block of KV data from VRAM to LDS.

        block_idx_const: Python int [0..8], which 64-col block.
        """
        lds_warp_offset = block_idx_const * KV_BLOCK_BYTES
        # p_lds_kv_warp points to warp's sub-block start.
        # Actual LDS target: p_lds_kv_warp + block*KV_BLOCK_BYTES - block*64
        lds_base_i32 = _std_arith.AddIOp(
            p_lds_kv_warp_i32,
            _raw(
                arith.constant(
                    lds_warp_offset - block_idx_const * KV_NUM_COLS, type=T.i32
                )
            ),
        ).result

        if check_boundary:
            neg_one = _raw(arith.constant(-1, type=T.i32))
            is_oob = _std_arith.CmpIOp(CmpIPredicate.eq, _raw(row_i32), neg_one).result
            # For OOB: write zero to LDS
            if_op = scf.IfOp(is_oob, [], has_else=True)
            with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                # Write zero via ds_write_b32 at lane's position
                zero_u32 = _raw(arith.constant(0, type=T.i32))
                lane_offset = _std_arith.MulIOp(
                    _raw(lane_idx_i32),
                    _raw(arith.constant(4, type=T.i32)),
                ).result
                lds_addr_zero = _std_arith.AddIOp(
                    lds_base_i32,
                    _std_arith.AddIOp(
                        _raw(arith.constant(block_idx_const * KV_NUM_COLS, type=T.i32)),
                        lane_offset,
                    ).result,
                ).result
                lds_addr_i64 = _std_arith.ExtUIOp(T.i64, lds_addr_zero).result
                lds_ptr = _inttoptr_lds(lds_addr_i64)
                llvm.StoreOp(zero_u32, lds_ptr, alignment=4)
                scf.YieldOp([])
            with ir.InsertionPoint(if_op.regions[1].blocks[0]):
                # Normal load
                voff = _std_arith.AddIOp(
                    _std_arith.MulIOp(
                        _raw(row_i32),
                        _raw(arith.constant(QK_HEAD_DIM, type=T.i32)),
                    ).result,
                    _raw(col_base_i32),
                ).result
                col_off = arith.constant(block_idx_const * KV_NUM_COLS, type=T.i32)
                lds_ptr_i64 = _std_arith.ExtUIOp(T.i64, lds_base_i32).result
                lds_ptr = _inttoptr_lds(lds_ptr_i64)
                rocdl.raw_ptr_buffer_load_lds(
                    kv_rsrc,
                    lds_ptr,
                    _raw(c_dword_sz),
                    voff,
                    _raw(c_aux_zero),
                    _raw(col_off),
                    _raw(c_aux_zero),
                )
                scf.YieldOp([])
        else:
            voff = _std_arith.AddIOp(
                _std_arith.MulIOp(
                    _raw(row_i32),
                    _raw(arith.constant(QK_HEAD_DIM, type=T.i32)),
                ).result,
                _raw(col_base_i32),
            ).result
            col_off = arith.constant(block_idx_const * KV_NUM_COLS, type=T.i32)
            lds_ptr_i64 = _std_arith.ExtUIOp(T.i64, lds_base_i32).result
            lds_ptr = _inttoptr_lds(lds_ptr_i64)
            rocdl.raw_ptr_buffer_load_lds(
                kv_rsrc,
                lds_ptr,
                _raw(c_dword_sz),
                voff,
                _raw(c_aux_zero),
                _raw(col_off),
                _raw(c_aux_zero),
            )

    def _async_load_kv_all(
        p_lds_kv_warp_i32, row_i32, col_base_i32, check_boundary=False
    ):
        """Load all 9 blocks of a KV tile."""
        for blk in range_constexpr(KV_NUM_BLOCKS):
            _async_load_k_tile(
                p_lds_kv_warp_i32,
                row_i32,
                col_base_i32,
                blk,
                check_boundary=check_boundary,
            )

    # ---- Inline-asm prefetch: fully opaque to LLVM waitcnt analysis ----
    def _prefetch_k_tile_asm(p_lds_kv_warp_i32, row_i32, col_base_i32, block_idx_const):
        """Prefetch one KV block via inline asm buffer_load_dword lds.

        Uses inline asm for BOTH the normal load AND the OOB zero-write
        so LLVM sees no LDS operations and won't insert spurious
        s_waitcnt vmcnt(0) before subsequent ds_read ops.

        Per-lane OOB check (row_i32 == -1): scf.IfOp for branching,
        but both branches use inline asm for LDS operations so LLVM
        can't see them.
        """
        lds_warp_offset = block_idx_const * KV_BLOCK_BYTES
        lds_base_i32 = _std_arith.AddIOp(
            p_lds_kv_warp_i32,
            _raw(
                arith.constant(
                    lds_warp_offset - block_idx_const * KV_NUM_COLS, type=T.i32
                )
            ),
        ).result

        neg_one = _raw(arith.constant(-1, type=T.i32))
        is_oob = _std_arith.CmpIOp(CmpIPredicate.eq, _raw(row_i32), neg_one).result

        if_op = scf.IfOp(is_oob, [], has_else=True)
        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            # OOB: write zero to LDS via inline asm ds_write_b32
            lane_offset = _std_arith.MulIOp(
                _raw(lane_idx_i32),
                _raw(arith.constant(4, type=T.i32)),
            ).result
            lds_zero_addr = _std_arith.AddIOp(
                lds_base_i32,
                _std_arith.AddIOp(
                    _raw(arith.constant(block_idx_const * KV_NUM_COLS, type=T.i32)),
                    lane_offset,
                ).result,
            ).result
            llvm.InlineAsmOp(
                res=None,
                operands_=[lds_zero_addr, _raw(arith.constant(0, type=T.i32))],
                asm_string="ds_write_b32 $0, $1",
                constraints="v,v",
                has_side_effects=True,
                is_align_stack=False,
            )
            scf.YieldOp([])
        with ir.InsertionPoint(if_op.regions[1].blocks[0]):
            # Normal: inline asm buffer_load_dword lds
            voff = _std_arith.AddIOp(
                _std_arith.MulIOp(
                    _raw(row_i32),
                    _raw(arith.constant(QK_HEAD_DIM, type=T.i32)),
                ).result,
                _raw(col_base_i32),
            ).result
            col_off_imm = block_idx_const * KV_NUM_COLS
            asm_str = (
                "s_mov_b32 m0, $0\n"
                "s_nop 0\n"
                f"buffer_load_dword $1, $2, 0 offen offset:{col_off_imm} lds"
            )
            llvm.InlineAsmOp(
                res=None,
                operands_=[lds_base_i32, voff, _raw(kv_rsrc)],
                asm_string=asm_str,
                constraints="s,v,s",
                has_side_effects=True,
                is_align_stack=False,
            )
            scf.YieldOp([])

    # ---- Helper: load K sub-tile from LDS (16x32 for MFMA) ----
    def _load_k_from_lds(p_lds_kv_base_idx, row_offset, col_offset):
        """Read 16x32 K sub-tile from LDS -> i64 for MFMA.

        row_offset: 0 or 16 (which half of BLOCK_N=32)
        col_offset: column offset in elements (multiple of 32)

        KvManagerV2 LDS address formula:
          row_phy = (row/2)*4 + (row%2)  where row = lane_idx % 16
          p = p_lds_kv + (row_phy/4)*KV_SUB_BYTES + (row_phy%4)*KV_BYTES_PER_ROW
              + (col%64)*sizeof(kv_t) + (col/64)*KV_BLOCK_BYTES
          fixed_offset = (row_offset/16)*2*KV_BYTES_PER_ROW
                       + (col_offset%64)*sizeof(kv_t)
                       + (col_offset/64)*KV_BLOCK_BYTES
        """
        row_in_mfma = lane_idx % arith.index(MFMA_M)
        # row_phy = (row/2)*4 + (row%2)
        row_phy = (row_in_mfma / arith.index(2)) * arith.index(
            4
        ) + row_in_mfma % arith.index(2)
        col_in_lane = (lane_idx / arith.index(MFMA_M)) * arith.index(MFMA_ELEM_PER_THR)

        # Dynamic part: based on lane position
        lds_lane_offset = (
            (row_phy / arith.index(4)) * arith.index(KV_SUB_BYTES)
            + (row_phy % arith.index(4)) * arith.index(KV_BYTES_PER_ROW)
            + (col_in_lane % arith.index(KV_NUM_COLS))
        )

        # Fixed part: based on compile-time row/col offsets
        fixed_offset = (
            (row_offset // 16) * 2 * KV_BYTES_PER_ROW
            + (col_offset % KV_NUM_COLS)
            + (col_offset // KV_NUM_COLS) * KV_BLOCK_BYTES
        )

        lds_addr = p_lds_kv_base_idx + lds_lane_offset + arith.index(fixed_offset)

        # ds_read_b64 -> 8 bytes = 1 i64 for MFMA input
        data = _lds_load(lds_addr, T.i64)
        return data

    # ---- Helper: load V from KV LDS (un-transposed) ----
    def _load_v_from_lds(p_lds_kv_base_idx, warp_idx_val, lane_idx_val):
        """Load un-transposed V: each warp reads 16x128 region.

        KvManagerV2::load_v_to_gpr pattern:
          row = (warp%2)*16 + lane/16*4
          row_phy = ((row%16)/2)*4 + 2*(row/16) + (row%2)
          col = (lane%16)*8 + (warp/2)*128
        Returns 8 i32 values.
        """
        row = (warp_idx_val % arith.index(2)) * arith.index(16) + (
            lane_idx_val / arith.index(16)
        ) * arith.index(4)
        row_mod16 = row % arith.index(16)
        row_phy = (
            (row_mod16 / arith.index(2)) * arith.index(4)
            + arith.index(2) * (row / arith.index(16))
            + row % arith.index(2)
        )
        col = (lane_idx_val % arith.index(16)) * arith.index(8) + (
            warp_idx_val / arith.index(2)
        ) * arith.index(128)

        lds_v_offset = (
            (row_phy / arith.index(4)) * arith.index(KV_SUB_BYTES)
            + (row_phy % arith.index(4)) * arith.index(KV_BYTES_PER_ROW)
            + (col / arith.index(KV_NUM_COLS)) * arith.index(KV_BLOCK_BYTES)
            + (col % arith.index(KV_NUM_COLS))
        )

        lds_addr = p_lds_kv_base_idx + lds_v_offset

        # 4 x ds_read_b64: load 8 dwords at strides matching KvManagerV2
        v_vals = []
        for pass_idx in range_constexpr(4):
            if pass_idx == 0:
                off = 0
            elif pass_idx == 1:
                off = KV_BYTES_PER_ROW
            elif pass_idx == 2:
                off = KV_SUB_BYTES
            else:
                off = KV_SUB_BYTES + KV_BYTES_PER_ROW
            data = _lds_load(
                lds_addr,
                T.i32x2,
                static_byte_offset=off,
            )
            v_vals.append(
                vector.extract(data, static_position=[0], dynamic_position=[])
            )
            v_vals.append(
                vector.extract(data, static_position=[1], dynamic_position=[])
            )
        return v_vals  # 8 i32 values

    # ---- Helper: transpose V in-register ----
    def _transpose_v(v8):
        """12x v_perm_b32 to transpose 4x8 fp8 block.

        Ported from VtManagerV1::transpose_v.
        Input:  v8[0..7] in row-major 4x8 layout
        Output: v8[0..7] in transposed layout for Vt storage
        """
        # Phase 1: perm_0 (c_perm0=0x05010400) and perm_3 (c_perm1=0x07030602)
        t0_0 = _vt_perm(v8[2], v8[0], c_perm0)
        t2_0 = _vt_perm(v8[2], v8[0], c_perm1)
        t0_1 = _vt_perm(v8[3], v8[1], c_perm0)
        t2_1 = _vt_perm(v8[3], v8[1], c_perm1)

        t1_0 = _vt_perm(v8[6], v8[4], c_perm0)
        t3_0 = _vt_perm(v8[6], v8[4], c_perm1)
        t1_1 = _vt_perm(v8[7], v8[5], c_perm0)
        t3_1 = _vt_perm(v8[7], v8[5], c_perm1)

        # Phase 2: perm_1 (c_perm2=0x05040100) and perm_2 (c_perm3=0x07060302)
        # Output order: r0_0, r0_1, r1_0, r1_1, r2_0, r2_1, r3_0, r3_1
        r = [None] * 8
        r[0] = _vt_perm(t1_0, t0_0, c_perm2)  # r0_0
        r[1] = _vt_perm(t1_1, t0_1, c_perm2)  # r0_1
        r[2] = _vt_perm(t1_0, t0_0, c_perm3)  # r1_0
        r[3] = _vt_perm(t1_1, t0_1, c_perm3)  # r1_1
        r[4] = _vt_perm(t3_0, t2_0, c_perm2)  # r2_0
        r[5] = _vt_perm(t3_1, t2_1, c_perm2)  # r2_1
        r[6] = _vt_perm(t3_0, t2_0, c_perm3)  # r3_0
        r[7] = _vt_perm(t3_1, t2_1, c_perm3)  # r3_1
        return r

    # ---- Helper: store transposed V to Vt LDS ----
    def _store_vt_to_lds(vt_lds_base_idx, warp_idx_val, lane_idx_val, vt8):
        """VtManagerV1::store_transposed_v_to_lds.

        4x8 block-wise row-major layout, no padding between rows/cols.
        row_blk = (warp%2)*4 + lane/16
        col_blk = (lane%16) + (warp/2)*16
        block_offset = (row_blk * VT_BLKS_PER_ROW_PAD + col_blk) * VT_ELEMS_PER_BLK
        """
        row_blk = (warp_idx_val % arith.index(2)) * arith.index(
            4
        ) + lane_idx_val / arith.index(16)
        col_blk = (lane_idx_val % arith.index(16)) + (
            warp_idx_val / arith.index(2)
        ) * arith.index(16)
        block_offset = (
            row_blk * arith.index(VT_BLKS_PER_ROW_PAD) + col_blk
        ) * arith.index(VT_ELEMS_PER_BLK)
        lds_vt_addr = vt_lds_base_idx + block_offset

        # ds_write_b128 x 2 (4 dwords each = 32 fp8)
        lo_packed = vector.from_elements(T.i32x4, vt8[0:4])
        lo_i8 = vector.bitcast(T.i8x16, lo_packed)
        vector.store(lo_i8, lds_buffer, [_raw(lds_vt_addr)])

        hi_packed = vector.from_elements(T.i32x4, vt8[4:8])
        hi_i8 = vector.bitcast(T.i8x16, hi_packed)
        vector.store(hi_i8, lds_buffer, [_raw(lds_vt_addr + arith.index(16))])

    # ---- Helper: load transposed V from Vt LDS ----
    def _load_vt_from_lds(vt_lds_base_idx, lane_idx_val, col_offset):
        """VtManagerV1::load_transposed_v_to_gpr.

        Each warp reads 32x16 block from Vt LDS. Returns 2 i32 via ds_read_b32x2.
        col_offset: Python int, multiple of 16, in [0, 512).

        LDS address formula:
          row_blk = lane/16
          col_blk = (lane%16) / VT_COLS_PER_THR
          row_inblk = lane % VT_ROWS_PER_THR
          col_inblk = ((lane%8) / VT_ROWS_PER_THR) * VT_ROWS_PER_THR
          fixed_col_blk = col_offset / VT_COLS_PER_THR
          offset_tl_bl = 4 * VT_BLKS_PER_ROW_PAD * VT_ELEMS_PER_BLK = 8448
        """
        fixed_col_blk = col_offset // VT_COLS_PER_THR
        fixed_block_offset = fixed_col_blk * VT_ELEMS_PER_BLK
        offset_tl_bl = 4 * VT_BLKS_PER_ROW_PAD * VT_ELEMS_PER_BLK  # 8448

        row_blk = lane_idx_val / arith.index(16)
        col_blk = (lane_idx_val % arith.index(16)) / arith.index(VT_COLS_PER_THR)
        row_inblk = lane_idx_val % arith.index(VT_ROWS_PER_THR)
        col_inblk = (
            (lane_idx_val % arith.index(8)) / arith.index(VT_ROWS_PER_THR)
        ) * arith.index(VT_ROWS_PER_THR)
        block_offset = (
            row_blk * arith.index(VT_BLKS_PER_ROW_PAD) + col_blk
        ) * arith.index(VT_ELEMS_PER_BLK)
        inblock_offset = row_inblk * arith.index(VT_COLS_PER_THR) + col_inblk

        lds_addr = vt_lds_base_idx + block_offset + inblock_offset

        # ds_read_b32 x 2
        v0 = _lds_load(lds_addr, T.i32, static_byte_offset=fixed_block_offset)
        v1 = _lds_load(
            lds_addr, T.i32, static_byte_offset=fixed_block_offset + offset_tl_bl
        )
        return v0, v1

    # ---- Helper: warp reduce (butterfly XOR) ----
    def _shfl_xor_f32(val_f32, offset_i32, width_i32):
        """XOR shuffle for f32 via bitcast to i32 and back."""
        # Bitcast f32 -> i32
        val_i32 = _std_arith.BitcastOp(T.i32, val_f32).result
        # Shuffle as i32
        peer_i32 = _mlir_gpu.ShuffleOp(
            val_i32, offset_i32, width_i32, _mlir_gpu.ShuffleMode.XOR
        ).shuffleResult
        # Bitcast i32 -> f32
        return _std_arith.BitcastOp(T.f32, peer_i32).result

    def _warp_reduce_max_16(val):
        """Butterfly max reduce across MFMA column groups.

        HK: reduce_range=64, stop_stride=15 -> strides [32, 16].
        This reduces across the 4 column groups (each owning 4 K positions)
        while keeping each row (Q head) independent.
        """
        w = _to_mlir(val)
        width = _std_arith.ConstantOp(
            T.i32, ir.IntegerAttr.get(T.i32, WARP_SIZE)
        ).result
        for sh in [32, 16]:
            offset = _std_arith.ConstantOp(T.i32, ir.IntegerAttr.get(T.i32, sh)).result
            peer = _shfl_xor_f32(w, offset, width)
            w = _std_arith.MaximumFOp(w, peer, fastmath=fm_no_inf).result
        return w

    def _warp_reduce_add_16(val):
        """Butterfly sum reduce across MFMA column groups."""
        w = _to_mlir(val)
        width = _std_arith.ConstantOp(
            T.i32, ir.IntegerAttr.get(T.i32, WARP_SIZE)
        ).result
        for sh in [32, 16]:
            offset = _std_arith.ConstantOp(T.i32, ir.IntegerAttr.get(T.i32, sh)).result
            peer = _shfl_xor_f32(w, offset, width)
            w = _std_arith.AddFOp(w, peer, fastmath=fm_fast).result
        return w

    # ---- Helper: Q loading (QManagerV3) ----
    def _load_q_to_regs(qo_start_i32):
        """Load Q from VRAM to registers via LDS staging.

        QManagerV3: each warp loads 16x64 per pass, 9 passes total.
        VRAM -> LDS (ds_write_b128), then LDS -> register (ds_read_b64).
        Returns (q_nope_regs, q_rope_regs):
          q_nope_regs: list of 16 v2i64 (16 sub-tiles x 32 cols each)
          q_rope_regs: list of 2 v2i64 (2 sub-tiles x 32 cols each)
        """
        p_lds_q_warp = (
            lds_base_idx
            + arith.index(P_LDS_Q)
            + warp_idx * arith.index(SZ_LDS_Q_PER_WARP)
        )

        # VRAM addressing: row = lane/4, col = (lane%4)*16
        # s_offset = warp * 16 * QK_HEAD_DIM * sizeof(fp8)
        # v_offset = (row * QK_HEAD_DIM + col) * sizeof(fp8)
        s_offset_i32 = _std_arith.MulIOp(
            _raw(warp_idx_i32),
            _raw(arith.constant(16 * QK_HEAD_DIM, type=T.i32)),
        ).result
        # Add qo_start offset: qo_start * NUM_QO_HEADS * QK_HEAD_DIM
        q_base_offset = _std_arith.MulIOp(
            _raw(qo_start_i32),
            _raw(arith.constant(NUM_QO_HEADS * QK_HEAD_DIM, type=T.i32)),
        ).result
        s_offset_i32 = _std_arith.AddIOp(s_offset_i32, q_base_offset).result

        row = lane_idx / arith.index(4)
        col = (lane_idx % arith.index(4)) * arith.index(16)
        v_offset_i32 = _index_cast_to_i32(row * arith.index(QK_HEAD_DIM) + col)

        # LDS store layout (QManagerV3):
        # row_st = lane/4, col_st = (lane%4)*16
        # v_offset_st = (row_st/2)*Q_BYTES_PER_2ROWS + ((row_st%2)*64 + col_st)
        row_st = lane_idx / arith.index(4)
        col_st = (lane_idx % arith.index(4)) * arith.index(16)
        lds_st_offset = (
            (row_st / arith.index(2)) * arith.index(Q_BYTES_PER_2ROWS)
            + (row_st % arith.index(2)) * arith.index(Q_ELEM_PER_ROW)
            + col_st
        )

        # LDS read layout (MFMA-compatible):
        # row_ld = lane%16, col_ld = (lane/16)*8
        # v_offset_ld = (row_ld/2)*Q_BYTES_PER_2ROWS + ((row_ld%2)*64 + col_ld)
        row_ld = lane_idx % arith.index(16)
        col_ld = (lane_idx / arith.index(16)) * arith.index(8)
        lds_ld_offset = (
            (row_ld / arith.index(2)) * arith.index(Q_BYTES_PER_2ROWS)
            + (row_ld % arith.index(2)) * arith.index(Q_ELEM_PER_ROW)
            + col_ld
        )

        q_regs = []  # Will hold 18 v2i64 = 16 nope + 2 rope

        # Fold s_offset and per-pass ioffset into voffset so that soffset=0.
        # LLVM ISel only extracts immediate offsets when soffset is literal 0.
        # v_offset_i32 is in bytes; buffer_load auto-scales by element_bytes
        # (i32 = 4), so divide by 4.  s_offset_i32 is also in bytes.
        voff_dw = _std_arith.DivSIOp(
            _std_arith.AddIOp(_raw(v_offset_i32), s_offset_i32).result,
            _raw(arith.constant(4, type=T.i32)),
        ).result

        # Pre-compute LDS pointers (constant across passes)
        lds_st_addr = p_lds_q_warp + lds_st_offset
        lds_st_i64 = arith.index_cast(T.i64, lds_st_addr)
        lds_st_ptr = _inttoptr_lds(_raw(lds_st_i64))
        lds_rd_addr = p_lds_q_warp + lds_ld_offset

        def _q_buf_load(pass_idx):
            voff_pass = _std_arith.AddIOp(
                voff_dw,
                _raw(arith.constant(pass_idx * Q_ELEM_PER_ROW // 4, type=T.i32)),
            ).result
            return buffer_ops.buffer_load(
                query_rsrc,
                voff_pass,
                vec_width=4,
                dtype=T.i32,
            )

        def _shuffle_q_through_lds(q_vram_data):
            """LDS write (ds_write_b128) + barrier + LDS read (2x ds_read_b64)."""
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            llvm.StoreOp(_raw(q_vram_data), lds_st_ptr, alignment=16)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            q0 = _lds_load(lds_rd_addr, T.i64, static_byte_offset=0)
            q1 = _lds_load(lds_rd_addr, T.i64, static_byte_offset=MFMA_K)
            return (q0, q1)

        # 3-deep pipeline: keep 2 buffer_loads in flight while shuffling
        # the completed one through LDS (matches HK QManagerV3).
        #   Before loop: issue passes 0, 1
        #   Iteration i: wait(1), issue pass i+2, shuffle pass i
        #   Last 2 iters: wait(0), shuffle (no new issue)
        loads = [None, None, None]
        loads[0] = _q_buf_load(0)
        loads[1] = _q_buf_load(1)

        for i in range_constexpr(9):
            slot = i % 3
            issue_pass = i + 2

            if issue_pass < 9:
                rocdl.s_waitcnt(_encode_waitcnt(vmcnt=1))
                loads[issue_pass % 3] = _q_buf_load(issue_pass)
            else:
                rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))

            q_regs.append(_shuffle_q_through_lds(loads[slot]))

        # Split into nope (passes 0-7 -> 16 sub-tiles) and rope (pass 8 -> 2 sub-tiles)
        q_nope_packs = []
        for i in range_constexpr(8):
            q_nope_packs.append(q_regs[i][0])  # sub-tile 0
            q_nope_packs.append(q_regs[i][1])  # sub-tile 1
        q_rope_packs = [q_regs[8][0], q_regs[8][1]]
        return q_nope_packs, q_rope_packs

    # ---- Helper: softmax scale + boundary masking ----
    def _softmax_scale_p(p_vals, col_0_start_i32, kv_end_i32, check_boundary):
        """Scale p_vals by softmax_scale, mask OOB to -inf."""
        result = [None] * 8
        for i in range_constexpr(8):
            result[i] = _std_arith.MulFOp(
                _raw(p_vals[i]), _raw(softmax_scale), fastmath=fm_fast
            ).result

        if check_boundary:
            for i in range_constexpr(8):
                # Position of this element: col_0_start + (i//4)*16 + (i%4)
                sub_offset = (i // 4) * 16 + (i % 4)
                pos_i32 = _std_arith.AddIOp(
                    _raw(col_0_start_i32),
                    _raw(arith.constant(sub_offset, type=T.i32)),
                ).result
                is_oob = _std_arith.CmpIOp(
                    CmpIPredicate.sge, pos_i32, _raw(kv_end_i32)
                ).result
                result[i] = _std_arith.SelectOp(
                    is_oob, _raw(c_neg_inf_f32), result[i]
                ).result
        return result

    # ---- Helper: online softmax ----
    def _softmax(
        p_vals,
        row_max_old,
        row_sum_e_old,
        is_first_iter,
        kv_tile_start_i32,
        kv_end_i32,
        check_boundary,
    ):
        """Online softmax: scale -> max -> exp2 -> sum -> rescale.

        p_vals: 8 f32 attention scores for this thread
        Returns: (p_exp_vals, row_max_new, row_sum_e_new, rescale)
        """
        # Column index for this thread's first element
        col_0_idx = lane_idx / arith.index(16)
        col_0_start_i32 = _std_arith.AddIOp(
            _raw(_index_cast_to_i32(col_0_idx * arith.index(4))),
            _raw(kv_tile_start_i32),
        ).result

        # Scale and mask
        scaled = _softmax_scale_p(p_vals, col_0_start_i32, kv_end_i32, check_boundary)

        # Local max of 8 values
        local_max = scaled[0]
        for i in range_constexpr(1, 8):
            local_max = _std_arith.MaximumFOp(
                local_max, _raw(scaled[i]), fastmath=fm_no_inf
            ).result

        # Warp reduce max (within 16-lane groups)
        local_max = _warp_reduce_max_16(local_max)

        # New row max
        if is_first_iter:
            new_row_max = local_max
            rescale = _raw(c_one_f32)
        else:
            new_row_max = _std_arith.MaximumFOp(
                local_max, _raw(row_max_old), fastmath=fm_no_inf
            ).result
            # rescale = exp2((old_max - new_max) * log2e)
            diff = _std_arith.SubFOp(
                _raw(row_max_old), new_row_max, fastmath=fm_no_inf
            ).result
            rescale_arg = _std_arith.MulFOp(
                diff, _raw(c_log2e), fastmath=fm_no_inf
            ).result
            rescale = _fast_exp2(rescale_arg)

        # exp(p - max) for each value, and sum
        p_exp_vals = [None] * 8
        local_sum = _raw(c_zero_f32)
        for i in range_constexpr(8):
            # exp2((p[i] - new_max) * log2e)
            diff = _std_arith.SubFOp(
                _raw(scaled[i]), new_row_max, fastmath=fm_no_inf
            ).result
            exp_arg = _std_arith.MulFOp(diff, _raw(c_log2e), fastmath=fm_no_inf).result
            p_exp_vals[i] = _fast_exp2(exp_arg)
            local_sum = _std_arith.AddFOp(
                local_sum, p_exp_vals[i], fastmath=fm_fast
            ).result

        # Warp reduce sum
        local_sum = _warp_reduce_add_16(local_sum)

        # Update row_sum_e
        if is_first_iter:
            row_sum_e_new = local_sum
        else:
            row_sum_e_new = _std_arith.AddFOp(
                _std_arith.MulFOp(
                    rescale, _raw(row_sum_e_old), fastmath=fm_fast
                ).result,
                local_sum,
                fastmath=fm_fast,
            ).result

        return p_exp_vals, new_row_max, row_sum_e_new, rescale

    # ---- Helper: pack P from f32 to fp8 ----
    def _pack_p_to_fp8(p_exp_vals):
        """Pack 8 f32 -> 2 i32 (4x cvt_pk_fp8_f32) -> 1 i64 for MFMA."""
        w0 = rocdl.cvt_pk_fp8_f32(
            T.i32, _raw(p_exp_vals[0]), _raw(p_exp_vals[1]), c_zero_i32, 0
        )
        w0 = rocdl.cvt_pk_fp8_f32(
            T.i32, _raw(p_exp_vals[2]), _raw(p_exp_vals[3]), w0, 1
        )
        w1 = rocdl.cvt_pk_fp8_f32(
            T.i32, _raw(p_exp_vals[4]), _raw(p_exp_vals[5]), c_zero_i32, 0
        )
        w1 = rocdl.cvt_pk_fp8_f32(
            T.i32, _raw(p_exp_vals[6]), _raw(p_exp_vals[7]), w1, 1
        )
        w0_i64 = _std_arith.ExtUIOp(T.i64, w0).result
        w1_i64 = _std_arith.ExtUIOp(T.i64, w1).result
        c32_i64 = _std_arith.ConstantOp(T.i64, ir.IntegerAttr.get(T.i64, 32)).result
        w1_shifted = _std_arith.ShLIOp(w1_i64, c32_i64).result
        p_pack = _std_arith.OrIOp(w0_i64, w1_shifted).result
        return p_pack

    # ---- Helper: rescale oaccu ----
    def _rescale_oaccu(oaccu, rescale):
        """Multiply all oaccu accumulators by rescale factor."""
        rescale_vec = vector.broadcast(T.f32x4, rescale)
        result = [None] * len(oaccu)
        for i in range_constexpr(len(oaccu)):
            result[i] = _std_arith.MulFOp(
                _raw(oaccu[i]), _raw(rescale_vec), fastmath=fm_fast
            ).result
        return result

    # ---- Helper: output bf16 (simplified direct write, no LDS reshape) ----
    def _write_output_bf16(oaccu, qo_start_i32, p_lds_o_base_idx):
        """Write normalized oaccu to final_output as bf16.

        Simplified version: each lane directly converts f32->bf16 and writes
        via buffer_store. MFMA layout: lane%16 = row (Q head),
        (lane/16)*4 + elem = col within sub-tile.
        """

        # MFMA layout: row = lane%16, col_base = (lane/16)*4
        mfma_row = lane_idx % arith.index(16)
        mfma_col_base = (lane_idx / arith.index(16)) * arith.index(4)
        qo_start_idx = arith.index_cast(T.index, qo_start_i32)
        # VRAM row = mfma_row + qo_start*128 + warp*16
        row_vram = (
            mfma_row
            + qo_start_idx * arith.index(NUM_QO_HEADS)
            + warp_idx * arith.index(16)
        )

        for tile_idx in range_constexpr(NUM_PV_ITERS * 2):
            col_offset = tile_idx * MFMA_N  # 0, 16, 32, ... 496

            # Convert v4f32 -> v4bf16
            bf16_vals = _std_arith.TruncFOp(T.bf16x4, _raw(oaccu[tile_idx])).result

            # Bitcast v4bf16 (8 bytes) -> v2i32 for buffer_store_dwordx2
            # Actually v4bf16 -> i64 -> use i64 store directly
            # Or: cast v4bf16 to v4i16, then to v2i32
            # Simplest: store v4bf16 as-is using f16 buffer format

            # Cast v4bf16 -> v4i16 -> v2i32 for buffer_store
            i16_vals = _std_arith.BitcastOp(T.i16x4, bf16_vals).result

            # Shuffle i16 pairs into i32: [i16_0, i16_1] -> i32_0, [i16_2, i16_3] -> i32_1
            i16_0 = vector.extract(i16_vals, static_position=[0], dynamic_position=[])
            i16_1 = vector.extract(i16_vals, static_position=[1], dynamic_position=[])
            i16_2 = vector.extract(i16_vals, static_position=[2], dynamic_position=[])
            i16_3 = vector.extract(i16_vals, static_position=[3], dynamic_position=[])

            # Pack i16 pairs to i32
            lo_0 = _std_arith.ExtUIOp(T.i32, i16_0).result
            hi_0 = _std_arith.ExtUIOp(T.i32, i16_1).result
            c16 = _raw(arith.constant(16, type=T.i32))
            dw0 = _std_arith.OrIOp(lo_0, _std_arith.ShLIOp(hi_0, c16).result).result

            lo_1 = _std_arith.ExtUIOp(T.i32, i16_2).result
            hi_1 = _std_arith.ExtUIOp(T.i32, i16_3).result
            dw1 = _std_arith.OrIOp(lo_1, _std_arith.ShLIOp(hi_1, c16).result).result

            # buffer_store_dwordx2 to final_output
            # Byte offset: (row * V_HEAD_DIM + col_offset + mfma_col_base) * sizeof(bf16)
            offset_i32 = _index_cast_to_i32(
                (
                    row_vram * arith.index(V_HEAD_DIM)
                    + mfma_col_base
                    + arith.index(col_offset)
                )
                * arith.index(2)
            )
            v2i32_vals = vector.from_elements(T.i32x2, [dw0, dw1])
            buffer_ops.buffer_store(
                v2i32_vals,
                final_output_rsrc,
                offset_i32,
                offset_is_bytes=True,
            )

    # ---- Helper: output f32 split (direct write, no LDS reshape) ----
    def _write_output_split(
        oaccu, partial_qo_loc_i32, row_max, row_sum_e, p_lds_o_base_idx
    ):
        """Write normalized oaccu to split_output as f32 + write LSE.

        Direct write: each lane writes its 4 f32 values per sub-tile
        using MFMA layout: row = lane%16, col_base = (lane/16)*4.
        """
        pqo_idx = arith.index_cast(T.index, partial_qo_loc_i32)
        mfma_row = lane_idx % arith.index(16)
        mfma_col_base = (lane_idx / arith.index(16)) * arith.index(4)
        row_vram = (
            mfma_row + pqo_idx * arith.index(NUM_QO_HEADS) + warp_idx * arith.index(16)
        )

        for tile_idx in range_constexpr(NUM_PV_ITERS * 2):
            col_offset = tile_idx * MFMA_N  # 0, 16, 32, ..., 496

            # buffer_store_dwordx4 to split_output
            # Byte offset: (row * V_HEAD_DIM + col_offset + mfma_col_base) * sizeof(f32)
            offset_i32 = _index_cast_to_i32(
                (
                    row_vram * arith.index(V_HEAD_DIM)
                    + mfma_col_base
                    + arith.index(col_offset)
                )
                * arith.index(4)  # sizeof(f32)
            )
            buffer_ops.buffer_store(
                oaccu[tile_idx],
                split_output_rsrc,
                offset_i32,
                offset_is_bytes=True,
            )

        # Write LSE: lse = row_max + ln(row_sum_e) / log2e
        # Only first 16 lanes per warp write (one per MFMA result row)
        if arith.cmpi(CmpIPredicate.ult, lane_idx_i32, arith.constant(16, type=T.i32)):
            # ln(sum_e) = log2(sum_e) * inv_log2e
            # math.log2 lowers to a single v_log_f32 instruction on AMD.
            log2_sum = _math.log2(_raw(row_sum_e))
            ln_sum = _std_arith.MulFOp(
                log2_sum, _raw(c_inv_log2e), fastmath=fm_fast
            ).result
            lse = _std_arith.AddFOp(
                _raw(row_max),
                ln_sum,
                fastmath=fm_fast,
            ).result
            row_idx_i32 = _std_arith.AddIOp(
                _std_arith.AddIOp(
                    _raw(lane_idx_i32),
                    _std_arith.MulIOp(
                        _raw(warp_idx_i32),
                        _raw(arith.constant(16, type=T.i32)),
                    ).result,
                ).result,
                _std_arith.MulIOp(
                    _raw(partial_qo_loc_i32),
                    _raw(arith.constant(NUM_QO_HEADS, type=T.i32)),
                ).result,
            ).result
            buffer_ops.buffer_store(
                lse,
                split_lse_rsrc,
                row_idx_i32,
            )

    # ---- Helper: process one KV tile (GEMM1 + softmax + V + GEMM2) ----
    # Interleaves async prefetch of the NEXT tile's KV data
    # into the GEMM1 NoPE loop (1 block per iteration, 9 total).
    def _process_tile(
        p_lds_kv_base,
        kv_tile_start_i32,
        kv_end_i32,
        q_nope,
        q_rope,
        row_max_in,
        row_sum_e_in,
        oaccu_in,
        is_first_iter,
        check_boundary,
        p_lds_kv_next_warp_i32=None,
        row_kv_ld_next=None,
        kv_ld_col_base_i32_arg=None,
    ):
        """Process one KV tile: QK GEMM -> softmax -> V transpose -> PV GEMM.

        When p_lds_kv_next_warp_i32 is provided, interleaves prefetch
        of the next tile's KV data during GEMM1 NoPE loop.
        Prefetch is always unconditional -- for the last tile, caller
        passes row_kv_ld_next=-1 which triggers OOB zeroing (harmless
        since that buffer won't be consumed).

        Returns (row_max, row_sum_e, oaccu).
        """
        do_prefetch = p_lds_kv_next_warp_i32 is not None

        def _maybe_prefetch(block_idx):
            """Issue prefetch unconditionally (OOB handled by row=-1)."""
            if not do_prefetch:
                return
            _prefetch_k_tile_asm(
                p_lds_kv_next_warp_i32,
                row_kv_ld_next,
                kv_ld_col_base_i32_arg,
                block_idx,
            )

        # ---- Prefetch block 0 of next tile (inline asm, opaque to LLVM) ----
        _maybe_prefetch(0)

        # ---- GEMM1: QK attention scores ----
        p_comp = [_raw(c_zero_v4f32), _raw(c_zero_v4f32)]

        for nope_pair in range_constexpr(NUM_NOPE_ITERS):
            tile_0 = nope_pair * 2
            tile_1 = nope_pair * 2 + 1

            k0_lo = _load_k_from_lds(p_lds_kv_base, 0, tile_0 * BLOCK_K)
            k0_hi = _load_k_from_lds(p_lds_kv_base, 16, tile_0 * BLOCK_K)
            k1_lo = _load_k_from_lds(p_lds_kv_base, 0, tile_1 * BLOCK_K)
            k1_hi = _load_k_from_lds(p_lds_kv_base, 16, tile_1 * BLOCK_K)

            # Prefetch block nope_pair+1 of next tile (inline asm)
            _maybe_prefetch(nope_pair + 1)

            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=2))

            q_0 = q_nope[tile_0]
            q_1 = q_nope[tile_1]

            if nope_pair == 0:
                p_comp[0] = _mfma_fp8(
                    T.f32x4, [k0_lo, q_0, _raw(c_zero_v4f32), 0, 0, 0]
                )
                p_comp[1] = _mfma_fp8(
                    T.f32x4, [k0_hi, q_0, _raw(c_zero_v4f32), 0, 0, 0]
                )
            else:
                p_comp[0] = _mfma_fp8(T.f32x4, [k0_lo, q_0, p_comp[0], 0, 0, 0])
                p_comp[1] = _mfma_fp8(T.f32x4, [k0_hi, q_0, p_comp[1], 0, 0, 0])

            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

            p_comp[0] = _mfma_fp8(T.f32x4, [k1_lo, q_1, p_comp[0], 0, 0, 0])
            p_comp[1] = _mfma_fp8(T.f32x4, [k1_hi, q_1, p_comp[1], 0, 0, 0])

        for rope_pair in range_constexpr(NUM_ROPE_ITERS):
            tile_0 = rope_pair * 2
            tile_1 = rope_pair * 2 + 1

            k0_lo = _load_k_from_lds(p_lds_kv_base, 0, (tile_0 + 16) * BLOCK_K)
            k0_hi = _load_k_from_lds(p_lds_kv_base, 16, (tile_0 + 16) * BLOCK_K)
            k1_lo = _load_k_from_lds(p_lds_kv_base, 0, (tile_1 + 16) * BLOCK_K)
            k1_hi = _load_k_from_lds(p_lds_kv_base, 16, (tile_1 + 16) * BLOCK_K)

            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=2))

            p_comp[0] = _mfma_fp8(T.f32x4, [k0_lo, q_rope[tile_0], p_comp[0], 0, 0, 0])
            p_comp[1] = _mfma_fp8(T.f32x4, [k0_hi, q_rope[tile_0], p_comp[1], 0, 0, 0])

            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

            p_comp[0] = _mfma_fp8(T.f32x4, [k1_lo, q_rope[tile_1], p_comp[0], 0, 0, 0])
            p_comp[1] = _mfma_fp8(T.f32x4, [k1_hi, q_rope[tile_1], p_comp[1], 0, 0, 0])

        # ---- Extract p_comp values for softmax ----
        p_vals = []
        for sub in range_constexpr(2):
            for ii in range_constexpr(4):
                p_vals.append(
                    vector.extract(
                        p_comp[sub], static_position=[ii], dynamic_position=[]
                    )
                )

        # ---- Load V from KV LDS ----
        v8_raw = _load_v_from_lds(p_lds_kv_base, warp_idx, lane_idx)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        rocdl.sched_barrier(0)

        # ---- Softmax ----
        p_exp_vals, row_max_new, row_sum_e_new, rescale = _softmax(
            p_vals,
            row_max_in,
            row_sum_e_in,
            is_first_iter,
            kv_tile_start_i32,
            kv_end_i32,
            check_boundary,
        )

        # ---- Transpose V and store to Vt LDS ----
        vt8 = _transpose_v(v8_raw)
        vt_lds_base = lds_base_idx + arith.index(P_LDS_VT)
        _store_vt_to_lds(vt_lds_base, warp_idx, lane_idx, vt8)

        # ---- Rescale oaccu (not first iter) ----
        if not is_first_iter:
            oaccu_in = _rescale_oaccu(oaccu_in, rescale)

        # ---- Pack P to fp8 ----
        p_pack = _pack_p_to_fp8(p_exp_vals)

        # ---- Barrier: wait for Vt store ----
        _barrier(lgkmcnt=0)
        rocdl.sched_barrier(0)

        # ---- GEMM2: PV accumulation ----
        c32_i64_pv = _std_arith.ConstantOp(T.i64, ir.IntegerAttr.get(T.i64, 32)).result
        oaccu_out = list(oaccu_in)
        for pv_iter in range_constexpr(NUM_PV_ITERS):
            col_offset_0 = pv_iter * MFMA_N * 2
            col_offset_1 = col_offset_0 + MFMA_N

            vt0_lo, vt0_hi = _load_vt_from_lds(vt_lds_base, lane_idx, col_offset_0)
            vt1_lo, vt1_hi = _load_vt_from_lds(vt_lds_base, lane_idx, col_offset_1)

            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=2))

            vt0_lo_i64 = _std_arith.ExtUIOp(T.i64, vt0_lo).result
            vt0_hi_i64 = _std_arith.ExtUIOp(T.i64, vt0_hi).result
            vt0_hi_shifted = _std_arith.ShLIOp(vt0_hi_i64, c32_i64_pv).result
            kv_mfma_0 = _std_arith.OrIOp(vt0_lo_i64, vt0_hi_shifted).result

            oaccu_out[pv_iter * 2] = _mfma_fp8(
                T.f32x4, [kv_mfma_0, p_pack, oaccu_out[pv_iter * 2], 0, 0, 0]
            )

            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

            vt1_lo_i64 = _std_arith.ExtUIOp(T.i64, vt1_lo).result
            vt1_hi_i64 = _std_arith.ExtUIOp(T.i64, vt1_hi).result
            vt1_hi_shifted = _std_arith.ShLIOp(vt1_hi_i64, c32_i64_pv).result
            kv_mfma_1 = _std_arith.OrIOp(vt1_lo_i64, vt1_hi_shifted).result

            oaccu_out[pv_iter * 2 + 1] = _mfma_fp8(
                T.f32x4, [kv_mfma_1, p_pack, oaccu_out[pv_iter * 2 + 1], 0, 0, 0]
            )

        return row_max_new, row_sum_e_new, oaccu_out

    # ==================================================================
    # Main kernel body: persistent-thread work loop
    # ==================================================================
    for work_idx in range(work_start_idx, work_end_idx):
        # Load MlaWorkInfo
        wi_base_i32 = _index_cast_to_i32(work_idx * SIZE_MLA_WORK_INFO_IN_DW)
        wi_dw1_4 = buffer_ops.buffer_load(
            work_info_set_rsrc,
            arith.addi(wi_base_i32, arith.constant(1, type=T.i32)),
            vec_width=4,
            dtype=T.i32,
        )
        wi_dw5 = buffer_ops.buffer_load(
            work_info_set_rsrc,
            arith.addi(wi_base_i32, arith.constant(5, type=T.i32)),
            vec_width=1,
            dtype=T.i32,
        )
        partial_qo_loc = rocdl.readfirstlane(T.i32, _raw(vector.extract(wi_dw1_4, [0])))
        qo_start = rocdl.readfirstlane(T.i32, _raw(vector.extract(wi_dw1_4, [1])))
        qo_end = rocdl.readfirstlane(T.i32, _raw(vector.extract(wi_dw1_4, [2])))
        kv_start = rocdl.readfirstlane(T.i32, _raw(vector.extract(wi_dw1_4, [3])))
        kv_end = rocdl.readfirstlane(T.i32, _raw(wi_dw5))
        kv_len = arith.subi(kv_end, kv_start)

        # ---- Load Q from VRAM to registers ----
        q_nope_packs, q_rope_packs = _load_q_to_regs(qo_start)
        rocdl.sched_barrier(0)

        # ---- KV tile iteration ----
        p_lds_kv_0_base = lds_base_idx + arith.index(P_LDS_KV_0)
        p_lds_kv_1_base = lds_base_idx + arith.index(P_LDS_KV_1)

        kv_warp_offset_i32 = _std_arith.MulIOp(
            _raw(warp_idx_i32),
            _raw(arith.constant(KV_SUB_BYTES, type=T.i32)),
        ).result

        p_lds_kv_0_warp_i32 = _std_arith.AddIOp(
            _raw(_index_cast_to_i32(p_lds_kv_0_base)),
            kv_warp_offset_i32,
        ).result
        p_lds_kv_1_warp_i32 = _std_arith.AddIOp(
            _raw(_index_cast_to_i32(p_lds_kv_1_base)),
            kv_warp_offset_i32,
        ).result

        # Initialize softmax state
        row_max = _raw(c_neg_inf_f32)
        row_sum_e = _raw(c_zero_f32)
        oaccu = [_raw(c_zero_v4f32)] * (NUM_PV_ITERS * 2)

        # Compute number of tiles
        c_block_n = arith.constant(BLOCK_N, type=T.i32)
        c_block_n_m1 = arith.constant(BLOCK_N - 1, type=T.i32)
        num_tiles = arith.divui(arith.addi(kv_len, c_block_n_m1), c_block_n)
        num_tiles_idx = arith.index_cast(T.index, num_tiles)

        # ---- Double-buffered tile loop ----
        # Buffer 0: first tile loaded synchronously
        # Prefetch of next tile interleaved in _process_tile during GEMM1
        # Subsequent tiles alternate buffers via tile_idx parity.

        # --- First tile: load all 9 blocks to KV_0 synchronously ---
        row_kv_ld_first = _get_kv_ld_row(kv_start, kv_end, True)
        _async_load_kv_all(
            p_lds_kv_0_warp_i32,
            row_kv_ld_first,
            kv_ld_col_base_i32,
            check_boundary=True,
        )

        _barrier(vmcnt=0, lgkmcnt=0)
        rocdl.sched_barrier(0)

        # Resolve row for tile 1 prefetch (raw, may be -1 for OOB)
        kv_start_plus_bn = _std_arith.AddIOp(
            _raw(kv_start),
            _raw(arith.constant(BLOCK_N, type=T.i32)),
        ).result
        row_kv_ld_tile1 = _get_kv_ld_row(kv_start_plus_bn, _raw(kv_end), True)

        # Process first tile -- always prefetch tile 1 to KV_1.
        # When num_tiles == 1, row_kv_ld_tile1 == -1 (OOB), which
        # triggers the OOB zeroing path in _prefetch_k_tile_asm.
        # Those zeros land in KV_1 which won't be consumed -- harmless.
        row_max, row_sum_e, oaccu = _process_tile(
            p_lds_kv_0_base,
            kv_start,
            kv_end,
            q_nope_packs,
            q_rope_packs,
            row_max,
            row_sum_e,
            oaccu,
            is_first_iter=True,
            check_boundary=True,
            p_lds_kv_next_warp_i32=p_lds_kv_1_warp_i32,
            row_kv_ld_next=row_kv_ld_tile1,
            kv_ld_col_base_i32_arg=kv_ld_col_base_i32,
        )

        # --- Remaining tiles [1, num_tiles) via scf.ForOp ---
        # Carried state: row_max, row_sum_e, oaccu[0..31]
        c_one_idx = arith.index(1)
        init_args = [row_max, row_sum_e] + oaccu
        init_args = [_raw(v) if not isinstance(v, ir.Value) else v for v in init_args]

        for_op = scf.ForOp(
            _raw(c_one_idx),
            _raw(num_tiles_idx),
            _raw(c_one_idx),
            init_args,
        )
        with ir.InsertionPoint(for_op.body):
            tile_iv = for_op.induction_variable  # index type
            tile_iv_i32 = _index_cast_to_i32(tile_iv)
            kv_tile_start_i32 = _std_arith.AddIOp(
                _raw(kv_start),
                _std_arith.MulIOp(tile_iv_i32, _raw(c_block_n)).result,
            ).result

            # Unpack carried state
            rm_carried = for_op.inner_iter_args[0]
            rse_carried = for_op.inner_iter_args[1]
            oaccu_carried = [
                for_op.inner_iter_args[2 + i] for i in range(NUM_PV_ITERS * 2)
            ]

            # Buffer parity: tile 0 used KV_0 and prefetched to KV_1.
            #   tile 1 (odd):  curr=KV_1, next=KV_0
            #   tile 2 (even): curr=KV_0, next=KV_1
            tile_parity = _std_arith.AndIOp(
                tile_iv_i32,
                _raw(arith.constant(1, type=T.i32)),
            ).result
            is_odd = _std_arith.CmpIOp(
                CmpIPredicate.ne,
                tile_parity,
                _raw(arith.constant(0, type=T.i32)),
            ).result
            curr_base_idx = _std_arith.SelectOp(
                is_odd,
                _raw(p_lds_kv_1_base),
                _raw(p_lds_kv_0_base),
            ).result
            next_warp = _std_arith.SelectOp(
                is_odd,
                p_lds_kv_0_warp_i32,
                p_lds_kv_1_warp_i32,
            ).result

            # Wait for previous prefetch, then process curr buffer
            _barrier(vmcnt=0, lgkmcnt=0)
            rocdl.sched_barrier(0)

            # Resolve row for next tile prefetch (may be -1 for OOB).
            kv_tile_next_start = _std_arith.AddIOp(
                kv_tile_start_i32,
                _raw(arith.constant(BLOCK_N, type=T.i32)),
            ).result
            row_kv_ld_next = _get_kv_ld_row(kv_tile_next_start, _raw(kv_end), True)

            rm_new, rse_new, oaccu_new = _process_tile(
                curr_base_idx,
                kv_tile_start_i32,
                _raw(kv_end),
                q_nope_packs,
                q_rope_packs,
                rm_carried,
                rse_carried,
                oaccu_carried,
                is_first_iter=False,
                check_boundary=True,
                p_lds_kv_next_warp_i32=next_warp,
                row_kv_ld_next=row_kv_ld_next,
                kv_ld_col_base_i32_arg=kv_ld_col_base_i32,
            )

            yield_vals = [rm_new, rse_new] + oaccu_new
            yield_vals = [
                _raw(v) if not isinstance(v, ir.Value) else v for v in yield_vals
            ]
            scf.YieldOp(yield_vals)

        # Unpack results from for loop
        row_max = for_op.results[0]
        row_sum_e = for_op.results[1]
        oaccu = [for_op.results[2 + i] for i in range(NUM_PV_ITERS * 2)]

        # ---- Normalize output: oaccu *= 1/row_sum_e ----
        reci_sum = _std_arith.DivFOp(
            _raw(c_one_f32), row_sum_e, fastmath=fm_fast
        ).result
        reci_vec = vector.broadcast(T.f32x4, reci_sum)
        for i in range_constexpr(len(oaccu)):
            oaccu[i] = _std_arith.MulFOp(
                oaccu[i], _raw(reci_vec), fastmath=fm_fast
            ).result

        # ---- Output dispatch ----
        p_lds_o = p_lds_kv_0_base

        if arith.cmpi(CmpIPredicate.slt, partial_qo_loc, arith.constant(0, type=T.i32)):
            _write_output_bf16(oaccu, qo_start, p_lds_o)
        else:
            _write_output_split(oaccu, partial_qo_loc, row_max, row_sum_e, p_lds_o)


# ---------------------------------------------------------------------------
# JIT launcher
# ---------------------------------------------------------------------------
@flyc.jit
def launch_mla_fwd_decode_h128_fp8_fp8(
    query: fx.Tensor,
    kv_buffer: fx.Tensor,
    kv_page_indices: fx.Tensor,
    work_indptr: fx.Tensor,
    work_info_set: fx.Tensor,
    final_output: fx.Tensor,
    split_output: fx.Tensor,
    split_lse: fx.Tensor,
    softmax_scale: fx.Float32,
    num_cus: fx.Constexpr,
    lds_size: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """JIT host function: configures grid/block and launches the kernel."""
    kn_mla_fwd_decode_h128_fp8_fp8(
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse,
        softmax_scale,
    ).launch(
        grid=(num_cus, 1, 1),
        block=(NUM_THREADS, 1, 1),
        smem=0,  # LDS is statically allocated via SmemAllocator
        stream=stream,
    )
