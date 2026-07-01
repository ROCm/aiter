# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Debug micro-kernel: compare MFMA QK^T vs scalar warp-shuffle QK^T.

Runs on one (batch=0, head=0) tile:
  - Loads K[BLOCK_N=64, D=128] into LDS
  - Loads Q[BLOCK_M=16, D=128] into LDS
  - Computes S_mfma[N=64, M=16] via mfma_f32_16x16x32_bf16 (K=32, 4 steps)
  - Computes S_scalar[N=64, M=16] via scalar dot product + warp shuffle
  - Writes both to global fp32 buffers
  - Compares on CPU against torch reference
"""

import math
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import gpu as mlir_gpu
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch
from flydsl.compiler.kernel_function import CompilationContext

from aiter.ops.flydsl.kernels.tensor_shim import STensor

BLOCK_N = 64
BLOCK_M = 16
WMMA_SIZE = 16
WARP_SIZE = 64
NUM_WARPS = 4
BLOCK_THREADS = NUM_WARPS * WARP_SIZE
D = 128
LOG2E = math.log2(math.e)

# LDS: k[64,128] + q[16,128]
_K_OFF = 0
_Q_OFF = BLOCK_N * D * 2  # 16384
_LDS = _Q_OFF + BLOCK_M * D * 2  # 20480

GPU_ARCH = get_rocm_arch()
alloc = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="dbg_smem")
alloc._align(alloc.ptr, 16)
alloc.ptr = _LDS


def _buf_load_bf16_raw(rsrc, byte_off):
    i32 = T.i32
    dw = arith.shrui(byte_off, arith.constant(2, type=i32))
    hi = arith.andi(
        arith.shrui(byte_off, arith.constant(1, type=i32)), arith.constant(1, type=i32)
    )
    raw = buffer_ops.buffer_load(rsrc, dw, vec_width=1, dtype=i32)
    return arith.shli(
        arith.andi(
            arith.shrui(raw, arith.muli(hi, arith.constant(16, type=i32))),
            arith.constant(0xFFFF, type=i32),
        ),
        arith.constant(16, type=i32),
    )


@flyc.kernel
def debug_qkt_kernel(
    k_ptr: fx.Tensor,  # [BLOCK_N, D] bf16
    q_ptr: fx.Tensor,  # [BLOCK_M, D] bf16
    s_mfma: fx.Tensor,  # [BLOCK_N, BLOCK_M] f32  output
    s_scalar: fx.Tensor,  # [BLOCK_N, BLOCK_M] f32  output
    stride_kn: Int32,
    stride_kd: Int32,
    stride_qm: Int32,
    stride_qd: Int32,
):
    tid = ArithValue(fx.thread_idx.x)
    i32 = T.i32
    f32 = T.f32

    wid = arith.divui(tid, arith.constant(WARP_SIZE, type=i32))
    lane = arith.remui(tid, arith.constant(WARP_SIZE, type=i32))
    n_warp = arith.muli(wid, arith.constant(WMMA_SIZE, type=i32))

    k_rsrc = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
    q_rsrc = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
    sm_rsrc = buffer_ops.create_buffer_resource(s_mfma, max_size=True)
    ss_rsrc = buffer_ops.create_buffer_resource(s_scalar, max_size=True)

    c2 = arith.constant(2, type=i32)
    c4 = arith.constant(4, type=i32)
    sn_k = ArithValue(stride_kn)
    sd_k = ArithValue(stride_kd)
    sm_q = ArithValue(stride_qm)
    sd_q = ArithValue(stride_qd)

    lds_base = alloc.get_base()
    k_smem = SmemPtr(lds_base, _K_OFF, T.bf16, shape=(BLOCK_N * D,))
    q_smem = SmemPtr(lds_base, _Q_OFF, T.bf16, shape=(BLOCK_M * D,))
    k_lds = STensor(k_smem, dtype=T.bf16, shape=(BLOCK_N * D,))
    q_lds = STensor(q_smem, dtype=T.bf16, shape=(BLOCK_M * D,))

    # ── Load K and Q into LDS ─────────────────────────────────────────────
    for ei in range_constexpr(BLOCK_N * D // BLOCK_THREADS):
        lin = arith.addi(tid, arith.constant(ei * BLOCK_THREADS, type=i32))
        row = arith.divui(lin, arith.constant(D, type=i32))
        col = arith.remui(lin, arith.constant(D, type=i32))
        boff = arith.muli(arith.addi(row * sn_k, col), c2)
        k_lds[arith.index_cast(T.index, lin)] = arith.bitcast(
            T.bf16,
            arith.trunci(
                T.i16,
                arith.shrui(
                    _buf_load_bf16_raw(k_rsrc, boff), arith.constant(16, type=i32)
                ),
            ),
        )

    for ei in range_constexpr(BLOCK_M * D // BLOCK_THREADS):
        lin = arith.addi(tid, arith.constant(ei * BLOCK_THREADS, type=i32))
        row = arith.divui(lin, arith.constant(D, type=i32))
        col = arith.remui(lin, arith.constant(D, type=i32))
        boff = arith.muli(arith.addi(row * sm_q, col), c2)
        q_lds[arith.index_cast(T.index, lin)] = arith.bitcast(
            T.bf16,
            arith.trunci(
                T.i16,
                arith.shrui(
                    _buf_load_bf16_raw(q_rsrc, boff), arith.constant(16, type=i32)
                ),
            ),
        )

    gpu.barrier()

    # ── Method A: MFMA (K=32, 4 steps) ───────────────────────────────────
    zero_v4 = arith.constant_vector(0.0, T.f32x4)
    s_acc = zero_v4
    kg32 = arith.divui(lane, arith.constant(WMMA_SIZE, type=i32))
    for d in range_constexpr(D // 32):
        # A: K_LDS[n_warp+l%16, d*32+(l//16)*8 .. +7]
        m_a = arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))
        row_a = arith.addi(n_warp, m_a)
        col_a = arith.addi(
            arith.constant(d * 32, type=i32),
            arith.muli(kg32, arith.constant(8, type=i32)),
        )
        idx_a = arith.index_cast(
            T.index, arith.addi(arith.muli(row_a, arith.constant(D, type=i32)), col_a)
        )
        a_frag = k_lds.vec_load((idx_a,), 8)  # v8bf16

        # B: Q_LDS[l%16, d*32+(l//16)*8 .. +7]  (row-major → gives K @ Q^T)
        n_b = arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))
        col_b = arith.addi(
            arith.constant(d * 32, type=i32),
            arith.muli(kg32, arith.constant(8, type=i32)),
        )
        idx_b = arith.index_cast(
            T.index, arith.addi(arith.muli(n_b, arith.constant(D, type=i32)), col_b)
        )
        b_frag = q_lds.vec_load((idx_b,), 8)  # v8bf16

        s_acc = rocdl.mfma_f32_16x16x32_bf16(T.f32x4, [a_frag, b_frag, s_acc, 0, 0, 0])

    # Write s_acc[4] using CORRECT C layout: C[(l//16)*4+ci, l%16]
    # Row = (l//16)*4+ci + n_warp (K-tile N-row), Col = l%16 (Q M-row)
    m_c = arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))  # l%16 = Q M-col
    kg_c = arith.divui(lane, arith.constant(WMMA_SIZE, type=i32))  # l//16
    for ci in range_constexpr(4):
        val = _llvm.extractelement(s_acc, arith.unwrap(arith.constant(ci, type=T.i32)))
        n_row_c = arith.addi(
            n_warp,
            arith.addi(
                arith.muli(kg_c, arith.constant(4, type=i32)),
                arith.constant(ci, type=i32),
            ),
        )  # n_warp+(l//16)*4+ci
        dw = arith.addi(arith.muli(n_row_c, arith.constant(BLOCK_M, type=i32)), m_c)
        buffer_ops.buffer_store(arith.bitcast(T.i32, val), sm_rsrc, dw)

    # ── Method B: scalar dot product + warp shuffle ───────────────────────
    # D=128, WARP_SIZE=64 → each lane covers D/WARP_SIZE=2 D-elements.
    # Lane accumulates both steps, then warp-reduces the sum.
    zero_f32 = arith.constant(0.0, type=f32)
    wsize = arith.constant(WARP_SIZE, type=i32)

    for n_off in range_constexpr(WMMA_SIZE):  # 16 N-rows per warp
        n_abs = arith.addi(n_warp, arith.constant(n_off, type=i32))

        for m_off in range_constexpr(BLOCK_M):
            # Accumulate over all D=128 elements: 2 per lane (lane covers d and d+64)
            partial = zero_f32
            for d_step in range_constexpr(D // WARP_SIZE):  # 2 steps
                d_elem = arith.addi(lane, arith.constant(d_step * WARP_SIZE, type=i32))
                k_bf16 = k_lds[
                    arith.index_cast(
                        T.index,
                        arith.addi(
                            arith.muli(n_abs, arith.constant(D, type=i32)), d_elem
                        ),
                    )
                ]
                q_bf16 = q_lds[
                    arith.index_cast(
                        T.index,
                        arith.addi(
                            arith.muli(
                                arith.constant(m_off, type=i32),
                                arith.constant(D, type=i32),
                            ),
                            d_elem,
                        ),
                    )
                ]
                partial = arith.addf(
                    partial,
                    arith.mulf(arith.extf(f32, k_bf16), arith.extf(f32, q_bf16)),
                )

            # Warp-reduce partial (sum over 64 lanes, each holding 2 D-elements)
            for sh in [WARP_SIZE >> s for s in range(1, WARP_SIZE.bit_length())]:
                off = arith.constant(sh, type=i32)
                peer = ArithValue(
                    mlir_gpu.ShuffleOp(partial, off, wsize, mode="xor").shuffleResult
                )
                partial = arith.addf(partial, peer)

            # Lane 0 of each warp writes
            is_lane0 = arith.cmpi(CmpIPredicate.eq, lane, arith.constant(0, type=i32))
            _if = scf.IfOp(is_lane0)
            with ir.InsertionPoint(_if.then_block):
                dw = arith.addi(
                    arith.muli(n_abs, arith.constant(BLOCK_M, type=i32)),
                    arith.constant(m_off, type=i32),
                )
                buffer_ops.buffer_store(arith.bitcast(T.i32, partial), ss_rsrc, dw)
                scf.YieldOp([])


@flyc.jit
def launch_debug_qkt(
    k,
    q,
    s_mfma,
    s_scalar,
    stride_kn,
    stride_kd,
    stride_qm,
    stride_qd,
    stream: fx.Stream = fx.Stream(None),
):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        alloc.finalized = False
        alloc.finalize()
    launcher = debug_qkt_kernel(
        k, q, s_mfma, s_scalar, stride_kn, stride_kd, stride_qm, stride_qd
    )
    launcher.launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)


# ── Run ───────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
K = torch.randn(BLOCK_N, D, dtype=torch.bfloat16, device="cuda")
Q = torch.randn(BLOCK_M, D, dtype=torch.bfloat16, device="cuda")

s_mfma_out = torch.zeros(BLOCK_N, BLOCK_M, dtype=torch.float32, device="cuda")
s_scalar_out = torch.zeros(BLOCK_N, BLOCK_M, dtype=torch.float32, device="cuda")

sn_k, sd_k = K.stride()
sm_q, sd_q = Q.stride()

launch_debug_qkt(K, Q, s_mfma_out, s_scalar_out, sn_k, sd_k, sm_q, sd_q)
torch.cuda.synchronize()

ref = K.float() @ Q.float().T  # [64, 16]

mfma_err = (s_mfma_out - ref).abs().max().item()
scalar_err = (s_scalar_out - ref).abs().max().item()

print(f"MFMA   max|S - ref| = {mfma_err:.4e}")
print(f"Scalar max|S - ref| = {scalar_err:.4e}")

if mfma_err < 0.5:
    print("MFMA  : CORRECT — QK^T MFMA is fine, bug is elsewhere")
else:
    print("MFMA  : WRONG   — MFMA QK^T is the bug")
    # Show first 4×4 corner to see the permutation pattern
    print("\nMFMA output s_mfma_out[:4, :4]:")
    print(s_mfma_out[:4, :4].cpu())
    print("Reference ref[:4, :4]:")
    print(ref[:4, :4].cpu())
    # Check if it might be transposed
    mfma_T_err = (s_mfma_out.T - ref).abs().max().item()
    print(
        f"\nmax|s_mfma^T - ref| = {mfma_T_err:.4e}  (checking if output is transposed)"
    )
    # Check for row/col permutation by comparing norms
    print("\nRow norms of MFMA vs ref:")
    print("MFMA row norms:", s_mfma_out.norm(dim=1)[:8].cpu().tolist())
    print("Ref  row norms:", ref.norm(dim=1)[:8].cpu().tolist())

if scalar_err < 0.5:
    print("Scalar: CORRECT — LDS loading is fine")
else:
    print("Scalar: WRONG   — LDS loading or scalar reduction is broken")
