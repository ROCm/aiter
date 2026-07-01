# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Debug micro-kernel: test K=16 MFMA with register-computed A-fragment.

Two variants of dQ = dS^T @ K:
  A) A-fragment from LDS (vec_load path — known to work for dK/dV)
  B) A-fragment built from register computation (insertelement path — the suspect)

If A works but B doesn't → the register-to-MFMA path is the bug.
If neither works → the dQ MFMA layout is wrong.
If both work → the bug is in how dS values are computed inline in fmha_bwd_dq_kernel.
"""

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch
from flydsl.compiler.kernel_function import CompilationContext

from aiter.ops.flydsl.kernels.tensor_shim import STensor

# ── Geometry (one K-group: 16 K-rows, 16 Q-rows, D=128) ────────────────────
K_ROWS = 16  # K-group size (one group of the 64-row K tile)
Q_ROWS = 16  # Q-rows per warp
D = 128
WMMA_SIZE = 16
A_FRAG = 4  # bf16 per lane for K=16 MFMA
D_TILES = D // WMMA_SIZE  # 8
C_FRAG = 4
NUM_WARPS = 1  # single warp for simplicity
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE  # 64

# LDS: k[16, 128] + ds[16, 16]  (only 1 warp so ds fits in per-warp region)
_K_LDS_OFF = 0
_DS_LDS_OFF = K_ROWS * D * 2  # 4096 bytes
_LDS_TOTAL = _DS_LDS_OFF + K_ROWS * Q_ROWS * 2  # 4096 + 512 = 4608 bytes

GPU_ARCH = get_rocm_arch()
alloc = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="dbg_dq_smem")
alloc._align(alloc.ptr, 16)
alloc.ptr = _LDS_TOTAL


def _mfma_k16(a_v4i16, b_v4i16, c_v4f32):
    return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a_v4i16, b_v4i16, c_v4f32, 0, 0, 0])


def _bf16_to_v4i16(v4bf16):
    return vector.bitcast(ir.VectorType.get([4], T.i16), v4bf16)


@flyc.kernel
def debug_dq_kernel(
    k_ptr: fx.Tensor,  # [K_ROWS, D] bf16
    ds_ptr: fx.Tensor,  # [K_ROWS, Q_ROWS] bf16  (dS, will build A via LDS)
    dq_lds: fx.Tensor,  # [Q_ROWS, D] f32  output via LDS A-frag path
    dq_reg: fx.Tensor,  # [Q_ROWS, D] f32  output via register A-frag path
    stride_kn: Int32,  # K row stride (elements)
    stride_qm: Int32,  # dS Q col stride (elements)
):
    tid = ArithValue(fx.thread_idx.x)
    i32 = T.i32
    f32 = T.f32

    lane = arith.remui(tid, arith.constant(WARP_SIZE, type=i32))  # 0..63
    kg = arith.divui(lane, arith.constant(WMMA_SIZE, type=i32))  # lane//16  0..3
    m_lane = arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))  # lane%16  0..15

    k_rsrc = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
    ds_rsrc = buffer_ops.create_buffer_resource(ds_ptr, max_size=True)
    ld_rsrc = buffer_ops.create_buffer_resource(dq_lds, max_size=True)
    rg_rsrc = buffer_ops.create_buffer_resource(dq_reg, max_size=True)

    c2 = arith.constant(2, type=i32)
    sn_k = ArithValue(stride_kn)
    sq_m = ArithValue(stride_qm)

    lds_base = alloc.get_base()
    k_smem = SmemPtr(lds_base, _K_LDS_OFF, T.bf16, shape=(K_ROWS * D,))
    ds_smem = SmemPtr(lds_base, _DS_LDS_OFF, T.bf16, shape=(K_ROWS * Q_ROWS,))
    K_LDS = STensor(k_smem, dtype=T.bf16, shape=(K_ROWS * D,))
    DS_LDS = STensor(ds_smem, dtype=T.bf16, shape=(K_ROWS * Q_ROWS,))

    # ── Load K[K_ROWS, D] into LDS ──────────────────────────────────────────
    # BLOCK_THREADS=64, K_ROWS*D = 16*128 = 2048 elements → 32 rounds
    for ei in range_constexpr(K_ROWS * D // BLOCK_THREADS):
        lin = arith.addi(tid, arith.constant(ei * BLOCK_THREADS, type=i32))
        row = arith.divui(lin, arith.constant(D, type=i32))
        col = arith.remui(lin, arith.constant(D, type=i32))
        dw = arith.shrui(
            arith.muli(arith.addi(row * sn_k, col), c2), arith.constant(2, type=i32)
        )  # dword offset
        raw = buffer_ops.buffer_load(k_rsrc, dw, vec_width=1, dtype=i32)
        hi = arith.andi(
            arith.shrui(
                arith.muli(arith.addi(row * sn_k, col), c2), arith.constant(1, type=i32)
            ),
            arith.constant(1, type=i32),
        )
        val_u = arith.shli(
            arith.andi(
                arith.shrui(raw, arith.muli(hi, arith.constant(16, type=i32))),
                arith.constant(0xFFFF, type=i32),
            ),
            arith.constant(16, type=i32),
        )
        K_LDS[arith.index_cast(T.index, lin)] = arith.bitcast(
            T.bf16,
            arith.trunci(T.i16, arith.shrui(val_u, arith.constant(16, type=i32))),
        )

    # ── Load dS[K_ROWS, Q_ROWS] into LDS (for the LDS A-frag path) ──────────
    # K_ROWS*Q_ROWS = 16*16 = 256 elements → 4 rounds
    for ei in range_constexpr(K_ROWS * Q_ROWS // BLOCK_THREADS):
        lin = arith.addi(tid, arith.constant(ei * BLOCK_THREADS, type=i32))
        row = arith.divui(lin, arith.constant(Q_ROWS, type=i32))  # K row
        col = arith.remui(lin, arith.constant(Q_ROWS, type=i32))  # Q col
        # dS is stored as [K_ROWS, Q_ROWS]: dS[k_row, q_col]
        dw = arith.shrui(
            arith.muli(arith.addi(row * arith.constant(Q_ROWS, type=i32), col), c2),
            arith.constant(2, type=i32),
        )
        raw = buffer_ops.buffer_load(ds_rsrc, dw, vec_width=1, dtype=i32)
        hi = arith.andi(
            arith.shrui(
                arith.muli(arith.addi(row * arith.constant(Q_ROWS, type=i32), col), c2),
                arith.constant(1, type=i32),
            ),
            arith.constant(1, type=i32),
        )
        val_u = arith.shli(
            arith.andi(
                arith.shrui(raw, arith.muli(hi, arith.constant(16, type=i32))),
                arith.constant(0xFFFF, type=i32),
            ),
            arith.constant(16, type=i32),
        )
        # Store dS^T to DS_LDS: DS_LDS[Q_ROWS_local=col, K_ROWS_local=row]
        # So DS_LDS[col * K_ROWS + row] = dS[row, col]
        ds_idx = arith.index_cast(
            T.index, arith.addi(arith.muli(col, arith.constant(K_ROWS, type=i32)), row)
        )
        DS_LDS[ds_idx] = arith.bitcast(
            T.bf16,
            arith.trunci(T.i16, arith.shrui(val_u, arith.constant(16, type=i32))),
        )

    gpu.barrier()

    # ── dS values for this lane (used in register path) ─────────────────────
    # For each ci: ds_val[ci] = dS[K_grp_row=(l//16)*4+ci, Q_col=l%16]
    # = DS_LDS (after transposing): DS_LDS[Q_col=l%16, K_row=(l//16)*4+ci]
    # = DS_LDS[m_lane * K_ROWS + kg*4 + ci]
    # We read BACK from DS_LDS to get the register values for the register path test.
    ds_vals = []
    for ci in range_constexpr(C_FRAG):
        ds_idx = arith.index_cast(
            T.index,
            arith.addi(
                arith.muli(m_lane, arith.constant(K_ROWS, type=i32)),
                arith.addi(
                    arith.muli(kg, arith.constant(C_FRAG, type=i32)),
                    arith.constant(ci, type=i32),
                ),
            ),
        )
        ds_vals.append(DS_LDS[ds_idx])  # scalar bf16

    # ── PATH A: LDS-based A-fragment ─────────────────────────────────────────
    # A[m=l%16, k=(l//16)*4..(l//16)*4+3] = DS_LDS^T[l%16, (l//16)*4..+3]
    # DS_LDS stores dS^T as [Q_ROWS=16, K_ROWS=16] with lds_cols=K_ROWS=16
    # Row = l%16 (Q-row), Col = (l//16)*4 (K-group offset within K_ROWS)
    col_a = arith.muli(kg, arith.constant(A_FRAG, type=i32))
    idx_a = arith.index_cast(
        T.index, arith.addi(arith.muli(m_lane, arith.constant(K_ROWS, type=i32)), col_a)
    )
    a_lds = _bf16_to_v4i16(DS_LDS.vec_load((idx_a,), A_FRAG))

    # ── PATH B: register-built A-fragment ─────────────────────────────────────
    # Same values as path A, but built via insertelement (register path)
    v4bf16_undef = ir.VectorType.get([4], T.bf16)
    a_reg_raw = _llvm.mlir_undef(v4bf16_undef)
    for ci in range_constexpr(C_FRAG):
        c_idx = arith.unwrap(arith.constant(ci, type=T.i32))
        a_reg_raw = _llvm.insertelement(a_reg_raw, ds_vals[ci], c_idx)
    a_reg = _bf16_to_v4i16(a_reg_raw)

    # ── dQ MFMA: both paths ──────────────────────────────────────────────────
    # B-fragment: K_LDS[K_row=(l//16)*4..+3, D_col=dt*16+l%16] (column access)
    zero_v4 = arith.constant_vector(0.0, T.f32x4)
    dq_lds_acc = [zero_v4] * D_TILES
    dq_reg_acc = [zero_v4] * D_TILES

    for dt in range_constexpr(D_TILES):
        # B from K_LDS: column access (K_row = (l//16)*4..+3, D_col = dt*16+l%16)
        n_col = arith.addi(arith.constant(dt * WMMA_SIZE, type=i32), m_lane)
        v4bf16_b = _llvm.mlir_undef(v4bf16_undef)
        for j in range_constexpr(A_FRAG):
            k_row = arith.addi(
                arith.muli(kg, arith.constant(A_FRAG, type=i32)),
                arith.constant(j, type=i32),
            )
            idx_b = arith.index_cast(
                T.index,
                arith.addi(arith.muli(k_row, arith.constant(D, type=i32)), n_col),
            )
            v4bf16_b = _llvm.insertelement(
                v4bf16_b, K_LDS[idx_b], arith.unwrap(arith.constant(j, type=T.i32))
            )
        b_frag = _bf16_to_v4i16(v4bf16_b)
        dq_lds_acc[dt] = _mfma_k16(a_lds, b_frag, dq_lds_acc[dt])
        dq_reg_acc[dt] = _mfma_k16(a_reg, b_frag, dq_reg_acc[dt])

    # ── Write results to global ──────────────────────────────────────────────
    # C[(l//16)*4+ci, l%16] = dQ[M_q=(l//16)*4+ci, D=dt*16+l%16]
    for dt in range_constexpr(D_TILES):
        for ci in range_constexpr(C_FRAG):
            val_l = ArithValue(
                _llvm.extractelement(
                    dq_lds_acc[dt], arith.unwrap(arith.constant(ci, type=T.i32))
                )
            )
            val_r = ArithValue(
                _llvm.extractelement(
                    dq_reg_acc[dt], arith.unwrap(arith.constant(ci, type=T.i32))
                )
            )
            m_q = arith.addi(
                arith.muli(kg, arith.constant(C_FRAG, type=i32)),
                arith.constant(ci, type=i32),
            )
            d_col = arith.addi(arith.constant(dt * WMMA_SIZE, type=i32), m_lane)
            dw = arith.addi(arith.muli(m_q, arith.constant(D, type=i32)), d_col)
            buffer_ops.buffer_store(arith.bitcast(T.i32, val_l), ld_rsrc, dw)
            buffer_ops.buffer_store(arith.bitcast(T.i32, val_r), rg_rsrc, dw)


@flyc.jit
def launch_debug_dq(
    k, ds, dq_lds, dq_reg, stride_kn, stride_qm, stream: fx.Stream = fx.Stream(None)
):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        alloc.finalized = False
        alloc.finalize()
    launcher = debug_dq_kernel(k, ds, dq_lds, dq_reg, stride_kn, stride_qm)
    launcher.launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)


# ── Run ───────────────────────────────────────────────────────────────────────
import math

torch.manual_seed(42)
K = torch.randn(K_ROWS, D, dtype=torch.bfloat16, device="cuda")
dS = torch.randn(K_ROWS, Q_ROWS, dtype=torch.bfloat16, device="cuda")

dq_lds_out = torch.zeros(Q_ROWS, D, dtype=torch.float32, device="cuda")
dq_reg_out = torch.zeros(Q_ROWS, D, dtype=torch.float32, device="cuda")

sn_k = K.stride(0)  # D
sq_m = 1  # dS stride (not used for loading here — we use Q_ROWS=16)

launch_debug_dq(K, dS, dq_lds_out, dq_reg_out, sn_k, sq_m)
torch.cuda.synchronize()

# Reference: dQ = dS^T @ K  (dS is [K_ROWS=16, Q_ROWS=16], K is [K_ROWS=16, D=128])
# dS^T is [Q_ROWS=16, K_ROWS=16], result is [Q_ROWS=16, D=128]
ref = dS.float().T @ K.float()  # [16, 128]

err_lds = (dq_lds_out - ref).abs().max().item()
err_reg = (dq_reg_out - ref).abs().max().item()

print(f"LDS  A-frag path: max|dQ - ref| = {err_lds:.4e}")
print(f"REG  A-frag path: max|dQ - ref| = {err_reg:.4e}")
print()

if err_lds < 0.5:
    print("LDS path: CORRECT — K=16 MFMA and B-frag are fine")
else:
    print(f"LDS path: WRONG (err={err_lds:.4e})")
    print("  dq_lds_out[:2, :4]:", dq_lds_out[:2, :4].cpu().tolist())
    print("  ref[:2, :4]:       ", ref[:2, :4].cpu().tolist())

if err_reg < 0.5:
    print("REG path: CORRECT — register-built A-frag works for MFMA")
else:
    print(f"REG path: WRONG (err={err_reg:.4e})")
    if err_lds < 0.5:
        print("  → Bug is in register A-frag construction; LDS path proves MFMA is OK")
        lds_match = (dq_lds_out - dq_reg_out).abs().max().item()
        print(f"  LDS vs REG diff: {lds_match:.4e}")
        print("  dq_reg_out[:2, :4]:", dq_reg_out[:2, :4].cpu().tolist())
        print("  dq_lds_out[:2, :4]:", dq_lds_out[:2, :4].cpu().tolist())
    else:
        print("  → Both paths fail; check MFMA layout or B-frag")

# Extra check: are both outputs zero?
print()
print(f"dq_lds max abs: {dq_lds_out.abs().max().item():.4e}")
print(f"dq_reg max abs: {dq_reg_out.abs().max().item():.4e}")
print(f"ref    max abs: {ref.abs().max().item():.4e}")
