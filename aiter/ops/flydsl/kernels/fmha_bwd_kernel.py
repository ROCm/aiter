# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Flash Attention Backward Main Kernel (non-causal, gfx950).

Algorithm — see fmha_bwd_preprocess.py for Pass 1.

Pass 2 (this kernel):
  Outer: one workgroup per K-row  →  grid=[B*H*sk, 1, 1]
  Block: HEAD_DIM threads          →  block=[D, 1, 1]
  Thread t owns D-element t of dK[k_row,t] and dV[k_row,t].

  Inner loop (dynamic, over Q-blocks of BLOCK_M rows each):
    1. Load Q[m, t] and dO[m, t] for m in [q_start .. q_start+BLOCK_M)
    2. qkT[m] = sum_d K[k_row,d]*Q[m,d]   — block-reduce via warp shuffle + LDS
    3. pT[m]  = exp2(qkT[m]*scale_log2e - lse[m])
    4. dV_acc[t] += sum_m pT[m]*dO[m,t]   — thread-local, no sync needed
    5. dpT[m] = sum_d V[k_row,d]*dO[m,d]  — block-reduce
    6. dsT[m] = pT[m]*(dpT[m] - delta[m])
    7. dK_acc[t] += sum_m dsT[m]*Q[m,t]   — thread-local
    8. dQ[m,t]   += dsT[m]*K[k_row,t]*scale  — global atomic add

  After loop: dK[k_row,t] = dK_acc[t]*scale; dV[k_row,t] = dV_acc[t]

Limitations (v1):
  - Non-causal only
  - MHA only (no GQA)
  - BHSD layout, contiguous tensors
  - seqlen_q must be a multiple of BLOCK_M
  - HEAD_DIM must be a multiple of WARP_SIZE
  - bf16/fp16 inputs; f32 dQ/dK/dV outputs
"""

from __future__ import annotations

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, buffer_ops, rocdl
from flydsl.expr import gpu as expr_gpu  # barrier()
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl._mlir import ir
from flydsl._mlir.dialects import gpu as mlir_gpu  # ShuffleOp
from flydsl._mlir.dialects import scf, llvm as _llvm
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch
from flydsl.compiler.kernel_function import CompilationContext

from .tensor_shim import STensor
from .kernels_common import get_warp_size

_LOG2E = math.log2(math.e)


def _lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def _buf_load_bf16_as_f32(rsrc, byte_off_i32):
    """Load one bf16 from global via buffer_load; return as f32."""
    i32 = T.i32
    f32 = T.f32
    dw_off = byte_off_i32 >> arith.constant(2, type=i32)
    is_hi = (byte_off_i32 >> arith.constant(1, type=i32)) & arith.constant(1, type=i32)
    raw = buffer_ops.buffer_load(rsrc, dw_off, vec_width=1, dtype=i32)
    half = (raw >> (is_hi * arith.constant(16, type=i32))) & arith.constant(
        0xFFFF, type=i32
    )
    return arith.bitcast(f32, half << arith.constant(16, type=i32))


def _buf_load_f32(rsrc, dw_off_i32):
    """Load one f32 from global at dword offset."""
    raw = buffer_ops.buffer_load(rsrc, dw_off_i32, vec_width=1, dtype=T.i32)
    return arith.bitcast(T.f32, raw)


def _buf_store_f32(val_f32, rsrc, dw_off_i32):
    """Store one f32 to global at dword offset."""
    buffer_ops.buffer_store(arith.bitcast(T.i32, val_f32), rsrc, dw_off_i32)


def _buf_atomic_add_f32(rsrc, dw_off_i32, val_f32):
    """Atomic float add to global at dword offset."""
    i32 = T.i32
    f32 = T.f32
    byte_off = dw_off_i32 << arith.constant(2, type=i32)
    zero = arith.constant(0, type=i32)
    _llvm.call_intrinsic(
        f32,
        "llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32",
        [val_f32, rsrc, byte_off, zero, zero],
        [],  # op_bundle_operands
        [],  # op_bundle_sizes
    )


def _wave_reduce_sum_f32(val, warp_size):
    """Intra-warp xor-shuffle sum reduction; returns wave-reduced value."""
    i32 = T.i32
    width = arith.constant(warp_size, type=i32)
    for sh in [warp_size >> s for s in range(1, warp_size.bit_length())]:
        off = arith.constant(sh, type=i32)
        peer = ArithValue(mlir_gpu.ShuffleOp(val, off, width, mode="xor").shuffleResult)
        val = arith.addf(val, peer)
    return val


def build_fmha_bwd_kernel_module(
    head_dim: int = 128,
    block_m: int = 16,
    dtype: str = "bf16",
):
    """Return a JIT launcher for the FMHA backward main kernel.

    Parameters
    ----------
    head_dim : int  Head dimension (multiple of WARP_SIZE).
    block_m  : int  Q-rows per inner iteration.
    dtype    : str  "bf16" or "fp16".
    """
    assert dtype in ("bf16", "fp16")
    WARP_SIZE = get_warp_size()
    GPU_ARCH = get_rocm_arch()
    assert head_dim % WARP_SIZE == 0
    NUM_WARPS = head_dim // WARP_SIZE
    BLOCK_THREADS = head_dim
    ELEM_BYTES = 2

    # LDS scratch: [NUM_WARPS * block_m] f32 — used for cross-warp reduction
    LDS_BYTES = NUM_WARPS * block_m * 4  # f32 per element

    allocator = SmemAllocator(
        None,
        arch=GPU_ARCH,
        global_sym_name=f"fmha_bwd_reduce_{head_dim}_{block_m}",
    )
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + LDS_BYTES

    @flyc.kernel
    def fmha_bwd_main_kernel(
        q_ptr: fx.Tensor,  # [B,H,Sq,D] bf16
        k_ptr: fx.Tensor,  # [B,H,Sk,D] bf16
        v_ptr: fx.Tensor,  # [B,H,Sk,D] bf16
        do_ptr: fx.Tensor,  # [B,H,Sq,D] bf16
        dq_ptr: fx.Tensor,  # [B,H,Sq,D] f32  (pre-zeroed)
        dk_ptr: fx.Tensor,  # [B,H,Sk,D] f32
        dv_ptr: fx.Tensor,  # [B,H,Sk,D] f32
        lse_ptr: fx.Tensor,  # [B,H,Sq]   f32
        delta_ptr: fx.Tensor,  # [B,H,Sq]   f32
        sm_scale: fx.Float32,
        # strides (elements) — all tensors BHSD contiguous
        stride_qb: Int32,
        stride_qh: Int32,
        stride_qm: Int32,
        stride_kb: Int32,
        stride_kh: Int32,
        stride_kn: Int32,
        stride_vb: Int32,
        stride_vh: Int32,
        stride_vn: Int32,
        stride_dob: Int32,
        stride_doh: Int32,
        stride_dom: Int32,
        stride_dqb: Int32,
        stride_dqh: Int32,
        stride_dqm: Int32,
        stride_dkb: Int32,
        stride_dkh: Int32,
        stride_dkn: Int32,
        stride_dvb: Int32,
        stride_dvh: Int32,
        stride_dvn: Int32,
        stride_lseb: Int32,
        stride_lseh: Int32,
        stride_lsem: Int32,
        stride_deltab: Int32,
        stride_deltah: Int32,
        stride_deltam: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        num_heads: Int32,
        batch: Int32,
    ):
        wg_id = ArithValue(fx.block_idx.x)  # one workgroup per K-row
        tid = ArithValue(fx.thread_idx.x)  # D-element index (0..head_dim-1)

        i32 = T.i32
        f32 = T.f32

        # Decode (batch, head, k_row)
        seqlen_k_v = ArithValue(seqlen_k)
        seqlen_q_v = ArithValue(seqlen_q)
        num_heads_v = ArithValue(num_heads)
        batch_v = ArithValue(batch)

        k_row = wg_id % seqlen_k_v
        tmp = wg_id // seqlen_k_v
        head_idx = tmp % num_heads_v
        batch_idx = tmp // num_heads_v

        # Guard: skip invalid workgroups
        valid = arith.cmpi(CmpIPredicate.ult, batch_idx, batch_v)
        _if_v = scf.IfOp(valid)
        with ir.InsertionPoint(_if_v.then_block):
            # LDS reduction scratch
            lds_base = allocator.get_base()
            lds_ptr = SmemPtr(lds_base, lds_offset, T.f32, shape=(NUM_WARPS * block_m,))
            lds = STensor(lds_ptr, dtype=T.f32, shape=(NUM_WARPS * block_m,))

            q_rsrc = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
            k_rsrc = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
            v_rsrc = buffer_ops.create_buffer_resource(v_ptr, max_size=True)
            do_rsrc = buffer_ops.create_buffer_resource(do_ptr, max_size=True)
            dq_rsrc = buffer_ops.create_buffer_resource(dq_ptr, max_size=True)
            dk_rsrc = buffer_ops.create_buffer_resource(dk_ptr, max_size=True)
            dv_rsrc = buffer_ops.create_buffer_resource(dv_ptr, max_size=True)
            lse_rsrc = buffer_ops.create_buffer_resource(lse_ptr, max_size=True)
            delta_rsrc = buffer_ops.create_buffer_resource(delta_ptr, max_size=True)

            c2 = arith.constant(ELEM_BYTES, type=i32)
            c4 = arith.constant(4, type=i32)
            zero = arith.constant(0.0, type=f32)

            sb_q = ArithValue(stride_qb)
            sh_q = ArithValue(stride_qh)
            sm_q = ArithValue(stride_qm)
            sb_k = ArithValue(stride_kb)
            sh_k = ArithValue(stride_kh)
            sn_k = ArithValue(stride_kn)
            sb_v = ArithValue(stride_vb)
            sh_v = ArithValue(stride_vh)
            sn_v = ArithValue(stride_vn)
            sb_do = ArithValue(stride_dob)
            sh_do = ArithValue(stride_doh)
            sm_do = ArithValue(stride_dom)
            sb_dq = ArithValue(stride_dqb)
            sh_dq = ArithValue(stride_dqh)
            sm_dq = ArithValue(stride_dqm)
            sb_dk = ArithValue(stride_dkb)
            sh_dk = ArithValue(stride_dkh)
            sn_dk = ArithValue(stride_dkn)
            sb_dv = ArithValue(stride_dvb)
            sh_dv = ArithValue(stride_dvh)
            sn_dv = ArithValue(stride_dvn)
            sb_l = ArithValue(stride_lseb)
            sh_l = ArithValue(stride_lseh)
            sm_l = ArithValue(stride_lsem)
            sb_d = ArithValue(stride_deltab)
            sh_d = ArithValue(stride_deltah)
            sm_d = ArithValue(stride_deltam)

            scale_v = ArithValue(sm_scale)
            scale_log2e = arith.constant(_LOG2E, type=f32) * scale_v

            lane_id = tid % arith.constant(WARP_SIZE, type=i32)
            wave_id = tid // arith.constant(WARP_SIZE, type=i32)

            # Load K[k_row, tid] and V[k_row, tid] — held throughout
            k_byte = (batch_idx * sb_k + head_idx * sh_k + k_row * sn_k + tid) * c2
            v_byte = (batch_idx * sb_v + head_idx * sh_v + k_row * sn_v + tid) * c2
            k_val = _buf_load_bf16_as_f32(k_rsrc, k_byte)
            v_val = _buf_load_bf16_as_f32(v_rsrc, v_byte)

            # dK and dV accumulators carried as iter_args across Q-blocks
            c_bm = arith.constant(block_m, type=i32)
            num_qblk_i32 = (seqlen_q_v + c_bm - arith.constant(1, type=i32)) // c_bm
            lb = arith.constant(0, type=T.index)
            ub = arith.index_cast(T.index, num_qblk_i32)
            step = arith.constant(1, type=T.index)

            for_op = scf.ForOp(lb, ub, step, [zero, zero])
            with ir.InsertionPoint(for_op.body):
                q_blk_idx = for_op.body.arguments[0]  # index
                dk_acc = for_op.body.arguments[1]  # f32 iter arg
                dv_acc = for_op.body.arguments[2]  # f32 iter arg

                q_blk_i32 = arith.index_cast(i32, q_blk_idx)
                q_start = q_blk_i32 * c_bm

                # ----------------------------------------------------------
                # Load Q[m, tid], dO[m, tid], lse[m], delta[m]
                # ----------------------------------------------------------
                q_vals = []
                do_vals = []
                lse_vals = []
                dlt_vals = []

                for m_off in range_constexpr(block_m):
                    m_idx = q_start + arith.constant(m_off, type=i32)
                    q_byte_m = (
                        batch_idx * sb_q + head_idx * sh_q + m_idx * sm_q + tid
                    ) * c2
                    do_byte_m = (
                        batch_idx * sb_do + head_idx * sh_do + m_idx * sm_do + tid
                    ) * c2
                    q_vals.append(_buf_load_bf16_as_f32(q_rsrc, q_byte_m))
                    do_vals.append(_buf_load_bf16_as_f32(do_rsrc, do_byte_m))

                    lse_dw = batch_idx * sb_l + head_idx * sh_l + m_idx * sm_l
                    dlt_dw = batch_idx * sb_d + head_idx * sh_d + m_idx * sm_d
                    lse_vals.append(_buf_load_f32(lse_rsrc, lse_dw))
                    dlt_vals.append(_buf_load_f32(delta_rsrc, dlt_dw))

                # ----------------------------------------------------------
                # Partial dot products: thread t contributes one D-element
                #   partial_qk[m] = K[k_row,t] * Q[m,t]
                #   partial_dp[m] = V[k_row,t] * dO[m,t]
                # ----------------------------------------------------------
                pqk = [arith.mulf(k_val, q_vals[m]) for m in range_constexpr(block_m)]
                pdp = [arith.mulf(v_val, do_vals[m]) for m in range_constexpr(block_m)]

                # Intra-warp reduction
                for m_off in range_constexpr(block_m):
                    pqk[m_off] = _wave_reduce_sum_f32(pqk[m_off], WARP_SIZE)
                    pdp[m_off] = _wave_reduce_sum_f32(pdp[m_off], WARP_SIZE)

                # Cross-warp: lane0 of each wave → LDS, barrier, all read
                is_lane0 = arith.cmpi(
                    CmpIPredicate.eq, lane_id, arith.constant(0, type=i32)
                )
                _if_l0 = scf.IfOp(is_lane0)
                with ir.InsertionPoint(_if_l0.then_block):
                    for m_off in range_constexpr(block_m):
                        lds_idx = arith.index_cast(
                            T.index,
                            wave_id * arith.constant(block_m, type=i32)
                            + arith.constant(m_off, type=i32),
                        )
                        lds[lds_idx] = pqk[m_off]
                    scf.YieldOp([])
                expr_gpu.barrier()

                qkT = []
                for m_off in range_constexpr(block_m):
                    acc_m = arith.constant(0.0, type=f32)
                    for w in range_constexpr(NUM_WARPS):
                        idx = arith.constant(w * block_m + m_off, type=T.index)
                        acc_m = arith.addf(acc_m, lds[idx])
                    qkT.append(acc_m)

                expr_gpu.barrier()

                # Reuse LDS for dpT
                _if_l0b = scf.IfOp(is_lane0)
                with ir.InsertionPoint(_if_l0b.then_block):
                    for m_off in range_constexpr(block_m):
                        lds_idx = arith.index_cast(
                            T.index,
                            wave_id * arith.constant(block_m, type=i32)
                            + arith.constant(m_off, type=i32),
                        )
                        lds[lds_idx] = pdp[m_off]
                    scf.YieldOp([])
                expr_gpu.barrier()

                dpT = []
                for m_off in range_constexpr(block_m):
                    acc_m = arith.constant(0.0, type=f32)
                    for w in range_constexpr(NUM_WARPS):
                        idx = arith.constant(w * block_m + m_off, type=T.index)
                        acc_m = arith.addf(acc_m, lds[idx])
                    dpT.append(acc_m)

                expr_gpu.barrier()

                # ----------------------------------------------------------
                # pT, dsT
                # ----------------------------------------------------------
                pT = []
                dsT = []
                log2e = arith.constant(_LOG2E, type=f32)
                for m_off in range_constexpr(block_m):
                    # lse is in natural-log domain; convert to log2 domain before subtracting
                    logit = qkT[m_off] * scale_log2e - lse_vals[m_off] * log2e
                    pt = rocdl.exp2(f32, logit)
                    pT.append(pt)
                    dsT.append(pt * (dpT[m_off] - dlt_vals[m_off]))

                # ----------------------------------------------------------
                # Thread-local dK and dV accumulation (no sync needed)
                # ----------------------------------------------------------
                new_dk = dk_acc
                new_dv = dv_acc
                for m_off in range_constexpr(block_m):
                    new_dv = arith.addf(new_dv, arith.mulf(pT[m_off], do_vals[m_off]))
                    new_dk = arith.addf(new_dk, arith.mulf(dsT[m_off], q_vals[m_off]))

                # ----------------------------------------------------------
                # dQ atomic add: dQ[m, tid] += dsT[m] * K[k_row, tid] * scale
                # ----------------------------------------------------------
                for m_off in range_constexpr(block_m):
                    m_idx = q_start + arith.constant(m_off, type=i32)
                    dq_dw = batch_idx * sb_dq + head_idx * sh_dq + m_idx * sm_dq + tid
                    contrib = dsT[m_off] * k_val * scale_v
                    _buf_atomic_add_f32(dq_rsrc, dq_dw, contrib)

                scf.YieldOp([new_dk, new_dv])

            # Retrieve final accumulated dk_acc, dv_acc
            dk_acc_final = for_op.results[0]
            dv_acc_final = for_op.results[1]

            # Write dK and dV (apply scale to dK once outside the loop)
            dk_dw = batch_idx * sb_dk + head_idx * sh_dk + k_row * sn_dk + tid
            dv_dw = batch_idx * sb_dv + head_idx * sh_dv + k_row * sn_dv + tid
            _buf_store_f32(dk_acc_final * scale_v, dk_rsrc, dk_dw)
            _buf_store_f32(dv_acc_final, dv_rsrc, dv_dw)

            scf.YieldOp([])

    @flyc.jit
    def launch_fmha_bwd_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        dq: fx.Tensor,
        dk: fx.Tensor,
        dv: fx.Tensor,
        lse: fx.Tensor,
        delta: fx.Tensor,
        sm_scale: fx.Float32,
        stride_qb: fx.Int32,
        stride_qh: fx.Int32,
        stride_qm: fx.Int32,
        stride_kb: fx.Int32,
        stride_kh: fx.Int32,
        stride_kn: fx.Int32,
        stride_vb: fx.Int32,
        stride_vh: fx.Int32,
        stride_vn: fx.Int32,
        stride_dob: fx.Int32,
        stride_doh: fx.Int32,
        stride_dom: fx.Int32,
        stride_dqb: fx.Int32,
        stride_dqh: fx.Int32,
        stride_dqm: fx.Int32,
        stride_dkb: fx.Int32,
        stride_dkh: fx.Int32,
        stride_dkn: fx.Int32,
        stride_dvb: fx.Int32,
        stride_dvh: fx.Int32,
        stride_dvn: fx.Int32,
        stride_lseb: fx.Int32,
        stride_lseh: fx.Int32,
        stride_lsem: fx.Int32,
        stride_deltab: fx.Int32,
        stride_deltah: fx.Int32,
        stride_deltam: fx.Int32,
        seqlen_q: fx.Int32,
        seqlen_k: fx.Int32,
        num_heads: fx.Int32,
        batch: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Emit the LDS global into the GPU module body
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalized = False
            allocator.finalize()

        total = arith.index_cast(T.index, seqlen_k * num_heads * batch)
        launcher = fmha_bwd_main_kernel(
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            lse,
            delta,
            sm_scale,
            stride_qb,
            stride_qh,
            stride_qm,
            stride_kb,
            stride_kh,
            stride_kn,
            stride_vb,
            stride_vh,
            stride_vn,
            stride_dob,
            stride_doh,
            stride_dom,
            stride_dqb,
            stride_dqh,
            stride_dqm,
            stride_dkb,
            stride_dkh,
            stride_dkn,
            stride_dvb,
            stride_dvh,
            stride_dvn,
            stride_lseb,
            stride_lseh,
            stride_lsem,
            stride_deltab,
            stride_deltah,
            stride_deltam,
            seqlen_q,
            seqlen_k,
            num_heads,
            batch,
        )
        launcher.launch(
            grid=(total, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fmha_bwd_kernel
