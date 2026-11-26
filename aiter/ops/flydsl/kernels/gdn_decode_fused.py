# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused GDN (gated delta rule) T=1 decode for gfx942, one workgroup per (token, v-head).

Per workgroup, in issue order:
  1. index load, then every lane issues its whole fp32 state slice (16 x dwordx4) up front,
  2. q/k/v channels computed from the in_proj output with the width-4 causal conv + SiLU
     and the indexed conv-state shift. q/k conv channels are shared by RATIO v-heads; the
     last of the RATIO workgroups to finish reading them writes the shifted q/k conv state back (atomic
     arrival counter, reset by that workgroup, so the counter is CUDA-graph safe),
  3. gating (softplus / exp decay, sigmoid beta rounded to the input dtype like the
     Triton kernel), q/k L2 norm,
  4. recurrent update h = h*exp(g); v' = (v - h.k)*beta; h += k v'^T; o = h.q,
  5. output: o (bf16, [B, 1, HV, V]) and, with WRITE_Z, a contiguous z [B, HV, V] for the
     caller's gated norm.
Pad slots (index < 0) write a zero output and leave both caches untouched.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from aiter.ops.flydsl.kernels.kernels_common import LOG2E, atomic_add_i32
from aiter.ops.flydsl.kernels.tensor_shim import (
    buf_base_i64,
    buf_copy_load,
    buf_copy_store,
    buf_scalar_load,
    ptr_buf_tensor,
)

_WARP = 64
# gfx942 buffer aux bits: the state is read and written back in place once per step.
# A store that hits a line the same kernel just read is slow on MI308X; streaming
# (nt) loads and stores of the state avoid most of that penalty (~20% at B=8).
_STATE_CACHE_POLICY = 2  # nt


def _ctr_add(counter_mem, idx):
    """Arrival-counter fetch-and-add, device scope; returns the previous count."""
    return atomic_add_i32(counter_mem, 1, idx, fx.rocdl.SyncScope.Agent)


def _ctr_reset(counter_mem, idx):
    fx.atomic_xchg(
        fx.get_iter(counter_mem) + idx, fx.Int32(0), syncscope=fx.rocdl.SyncScope.Agent
    )


def _dpp_f32(src, ctrl: int):
    """Row-local DPP move (no LDS): quad_perm / row_mirror / row_half_mirror."""
    i = fx.Float32(src).bitcast(fx.Int32)
    return fx.Int32(update_dpp_i32(i, i, ctrl)).bitcast(fx.Float32)


# all-reduce sum over aligned groups of 2/4/8/16 lanes
_DPP_STEPS = {1: 177, 2: 78, 4: 0x141, 8: 0x140}  # xor1, xor2, half_mirror, mirror


def _group_allreduce(x, group: int):
    o = 1
    while o < group:
        x = fx.Float32(x) + _dpp_f32(x, _DPP_STEPS[o])
        o *= 2
    return x


def _addr(mem, byte_offset):
    """i64 device address of a kernel tensor argument plus ``byte_offset``."""
    return buf_base_i64(mem) + fx.Int64(byte_offset)


def _sload_i32(mem, elem_idx):
    """Uniform int32 element load through the scalar unit (s_buffer_load_dword)."""
    return fx.Int32(buf_scalar_load(ptr_buf_tensor(mem, fx.Int32), elem_idx))


def _state_buf(addr_i64, num_records_bytes=None):
    """fp32 view for 16-byte state accesses at element offsets."""
    return ptr_buf_tensor(
        addr_i64,
        fx.Float32,
        unit_elems=4,
        unit_stride=1,
        num_records_bytes=num_records_bytes,
    )


def _sload_f32(mem, elem_idx):
    return _sload_i32(mem, elem_idx).bitcast(fx.Float32)


def _sload_bf16_as_f32(mem, elem_idx):
    """Uniform bf16 element: load its aligned dword and pick the half."""
    e = fx.Int32(elem_idx)
    w = _sload_i32(mem, e // fx.Int32(2))
    hi = (e % fx.Int32(2)) == fx.Int32(1)
    bits = hi.select(w & fx.Int32(-65536), w << fx.Int32(16))
    return bits.bitcast(fx.Float32)


def _vsum(v):
    return fx.Vector(v).reduce(fx.ReductionOp.ADD)


def _fma(a, b, c):
    return fx.math.fma(a, b, c)


def _sigmoid(x):
    return fx.Float32(1.0) / (fx.Float32(1.0) + fx.math.exp2(-x * fx.Float32(LOG2E)))


def _conv_silu(svals, xv, w):
    """Width-4 causal conv + SiLU of one channel, bf16-rounded like the Triton kernel."""
    w = fx.Vector(w)
    acc = fx.Float32(0.0)
    for j in range_constexpr(3):
        acc = acc + (svals[j] * w[j]).to(fx.BFloat16).to(fx.Float32)
    acc = acc + (xv * w[3]).to(fx.BFloat16).to(fx.Float32)
    return (acc * _sigmoid(acc)).to(fx.BFloat16).to(fx.Float32)


@functools.cache
def create_gdn_decode_fused_kernel(
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int = 128,
    dt_bias_dtype: str = "bf16",
    warp_threads_k: int = 8,
    softplus_threshold: float = 20.0,
    write_z: bool = False,
):
    H, HV, D = num_k_heads, num_v_heads, head_dim
    RATIO = HV // H
    assert HV % H == 0 and D == 128
    VPT = 4  # f32 values per lane per state transaction (16 B)
    num_warps = 4  # one thread per q/v channel plus one per k channel: NT = 2 * D
    NT = num_warps * _WARP
    WTK = warp_threads_k
    WTV = _WARP // WTK
    TILE_K = WTK * VPT
    K_ITERS = D // TILE_K
    V_GROUP = num_warps * WTV
    V_ITERS = D // V_GROUP
    assert D % TILE_K == 0 and D % V_GROUP == 0 and NT == 2 * D
    QK_DIM = 2 * H * D
    SCALE = D**-0.5
    dt_bias_is_bf16 = dt_bias_dtype == "bf16"

    name = (f"gdn_decode_fused_h{H}x{HV}_d{D}_{num_warps}w{WTV}x{WTK}") + (
        "_z" if write_z else ""
    )
    assert WTK <= 16

    @fx.struct
    class SharedStorage:
        qk: fx.Array[fx.Float32, 2 * D, 16]
        v: fx.Array[fx.Float32, 2 * D, 16]
        flag: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        x_mem: fx.Tensor,  # in_proj qkvz [B, *] = [q | k | v | z]
        ba_mem: fx.Tensor,  # in_proj ba [B, *] = [b(HV) | a(HV)]
        conv_w_mem: fx.Tensor,  # [conv_dim, 4] bf16
        conv_state_mem: fx.Tensor,  # conv state pool
        counter_mem: fx.Tensor,  # int32 [>= B*H] zero-initialised
        A_log_mem: fx.Tensor,
        dt_bias_mem: fx.Tensor,
        state_mem: fx.Tensor,  # fp32 ssm state pool, [slot, HV, V, K] inner contiguous
        indices_mem: fx.Tensor,  # int32 [B]
        out_mem: fx.Tensor,  # bf16
        z_out_mem: fx.Tensor,  # bf16 [B, HV, V] contiguous (write_z)
        stride_x_tok: fx.Int32,
        stride_ba_tok: fx.Int32,
        stride_conv_slot: fx.Int64,
        stride_conv_ch: fx.Int32,
        stride_conv_w: fx.Int32,
        stride_state_slot: fx.Int64,
        stride_out_tok: fx.Int32,
    ):
        x = ptr_buf_tensor(x_mem, fx.BFloat16)
        conv_w = ptr_buf_tensor(conv_w_mem, fx.BFloat16, unit_elems=4)
        out = ptr_buf_tensor(out_mem, fx.BFloat16)

        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        qk_lds = shared.qk.ptr
        q_lds = qk_lds
        k_lds = qk_lds + fx.Int32(D)  # q at [0, D), k at [D, 2D)
        v_lds = shared.v.ptr
        flag_lds = shared.flag.ptr

        tid = fx.thread_idx.x
        block = fx.block_idx.x
        b_i = block // fx.Int32(HV)
        hv = block % fx.Int32(HV)
        hk = hv // fx.Int32(RATIO)
        lane = tid % fx.Int32(_WARP)
        warp = tid // fx.Int32(_WARP)
        lane_k = lane % fx.Int32(WTK)
        k_vec_start = lane_k * fx.Int32(VPT)
        v_start = warp * fx.Int32(WTV) + lane // fx.Int32(WTK)

        # ---- metadata (uniform; scalar unit) ---------------------------------------
        b_off = b_i * stride_ba_tok + hv
        slot_raw = _sload_i32(indices_mem, b_i)
        r_b = _sload_bf16_as_f32(ba_mem, b_off)
        r_a = _sload_bf16_as_f32(ba_mem, b_off + fx.Int32(HV))
        r_A_log = _sload_f32(A_log_mem, hv)
        if fx.const_expr(dt_bias_is_bf16):
            r_dt = _sload_bf16_as_f32(dt_bias_mem, hv)
        else:
            r_dt = _sload_f32(dt_bias_mem, hv)
        valid = slot_raw >= fx.Int32(0)
        # pad slots read slot 0 (discarded) so every load stays in straight-line code
        slot64 = fx.Int64(valid.select(slot_raw, fx.Int32(0)))

        # ---- per-thread channel loads ----------------------------------------------
        # thread t < D owns q channel t and v channel t; D <= t < 2D owns k channel t-D.
        is_qv = tid < fx.Int32(D)
        ch_local = is_qv.select(tid, tid - fx.Int32(D))
        qk_ch = is_qv.select(
            hk * fx.Int32(D) + ch_local, fx.Int32(H * D) + hk * fx.Int32(D) + ch_local
        )
        v_ch = fx.Int32(QK_DIM) + hv * fx.Int32(D) + ch_local
        row_x = b_i * stride_x_tok
        x_qk = fx.Float32(x[row_x + qk_ch])
        x_v = fx.Float32(x[row_x + v_ch])
        wq = buf_copy_load(conv_w, qk_ch, fx.BFloat16, 4).to(fx.Float32)
        wv = buf_copy_load(conv_w, v_ch, fx.BFloat16, 4).to(fx.Float32)
        cs = ptr_buf_tensor(
            _addr(conv_state_mem, slot64 * stride_conv_slot * fx.Int64(2)), fx.BFloat16
        )
        cqk = qk_ch * stride_conv_ch
        cv = v_ch * stride_conv_ch
        s_qk = [fx.Float32(cs[cqk + fx.Int32(j) * stride_conv_w]) for j in range(3)]
        s_v = [fx.Float32(cs[cv + fx.Int32(j) * stride_conv_w]) for j in range(3)]
        if fx.const_expr(write_z):
            z_ch = fx.Int32(QK_DIM + HV * D) + hv * fx.Int32(D) + ch_local
            r_z_bf = fx.BFloat16(x[row_x + z_ch])
            # z rides along so the caller's gated norm reads a contiguous [B, HV, V]
            z_out = ptr_buf_tensor(z_out_mem, fx.BFloat16)
            if is_qv:
                z_out[b_i * fx.Int32(HV * D) + hv * fx.Int32(D) + ch_local] = r_z_bf

        # ---- state slice: all 16 transactions before anything waits ----------------
        st_base = slot64 * stride_state_slot + fx.Int64(hv) * fx.Int64(D * D)
        st_buf = _state_buf(_addr(state_mem, st_base * fx.Int64(4)))
        state_vecs = []
        for vi in range_constexpr(V_ITERS):
            gv = v_start + fx.Int32(vi * V_GROUP)
            for ki in range_constexpr(K_ITERS):
                kb = k_vec_start + fx.Int32(ki * TILE_K)
                state_vecs.append(
                    buf_copy_load(
                        st_buf,
                        gv * fx.Int32(D) + kb,
                        fx.Float32,
                        VPT,
                        cache_modifier=_STATE_CACHE_POLICY,
                    )
                )

        # ---- conv and q/k/v exchange through LDS -------------------------

        q_or_k = _conv_silu(s_qk, x_qk, wq)
        vv = _conv_silu(s_v, x_v, wv)
        fx.ptr_store(q_or_k, qk_lds + tid)
        # threads >= D store a duplicate into the pad half: no branch around the load
        fx.ptr_store(vv, v_lds + tid)

        # v conv channels are owned by this workgroup: shift now.
        if is_qv & valid:
            cs[cv] = s_v[1].to(fx.BFloat16)
            cs[cv + stride_conv_w] = s_v[2].to(fx.BFloat16)
            cs[cv + fx.Int32(2) * stride_conv_w] = x_v.to(fx.BFloat16)
        # LDS exchange barrier. Every thread has consumed its conv-state reads by now,
        # so the arrival counter can be bumped right after it.
        fx.gpu.barrier()
        # q/k conv channels are shared by RATIO workgroups: the last reader shifts
        # them. The atomic's return value is consumed only at the end (before the
        # epilogue barrier), so its latency hides behind the recurrence.
        ctr_idx = b_i * fx.Int32(H) + hk
        seen = fx.Int32(0)
        if (tid == fx.Int32(0)) & valid:
            seen = fx.Int32(_ctr_add(counter_mem, ctr_idx))

        # ---- gating -----------------------------------------------------------------
        xg = r_a + r_dt
        softplus = (xg <= fx.Float32(softplus_threshold)).select(
            fx.math.log(fx.Float32(1.0) + fx.math.exp2(xg * fx.Float32(LOG2E))), xg
        )
        g = -fx.math.exp2(r_A_log * fx.Float32(LOG2E)) * softplus
        decay = fx.math.exp2(g * fx.Float32(LOG2E))
        beta = _sigmoid(r_b).to(fx.BFloat16).to(fx.Float32)

        # ---- q/k fragments + l2 norm -------------------------------------------------
        vec_f32 = T.vec(VPT, T.f32)
        q_vecs, k_vecs = [], []
        sq = fx.Float32(0.0)
        sk = fx.Float32(0.0)
        for ki in range_constexpr(K_ITERS):
            kb = k_vec_start + fx.Int32(ki * TILE_K)
            qv = fx.ptr_load(q_lds + kb, result_type=vec_f32)
            kv = fx.ptr_load(k_lds + kb, result_type=vec_f32)
            q_vecs.append(qv)
            k_vecs.append(kv)
            sq = sq + _vsum(qv * qv)
            sk = sk + _vsum(kv * kv)
        sq = _group_allreduce(sq, WTK)
        sk = _group_allreduce(sk, WTK)
        inv_q = fx.Float32(1.0) / fx.math.sqrt(fx.Float32(sq) + fx.Float32(1e-6))
        inv_k = fx.Float32(1.0) / fx.math.sqrt(fx.Float32(sk) + fx.Float32(1e-6))
        for ki in range_constexpr(K_ITERS):
            q_vecs[ki] = q_vecs[ki] * fx.Float32(inv_q) * fx.Float32(SCALE)
            k_vecs[ki] = k_vecs[ki] * fx.Float32(inv_k)
        dkq = fx.full(VPT, 0.0, fx.Float32)
        for ki in range_constexpr(K_ITERS):
            dkq = _fma(k_vecs[ki], q_vecs[ki], dkq)
        dot_kq = _group_allreduce(_vsum(dkq), WTK)

        # ---- recurrence ----------------------------------------------------------------
        for vi in range_constexpr(V_ITERS):
            gv = v_start + fx.Int32(vi * V_GROUP)
            shk = fx.full(VPT, 0.0, fx.Float32)
            shq = fx.full(VPT, 0.0, fx.Float32)
            for ki in range_constexpr(K_ITERS):
                p = vi * K_ITERS + ki
                h = state_vecs[p] * fx.Float32(decay)
                state_vecs[p] = h
                shk = _fma(h, k_vecs[ki], shk)
                shq = _fma(h, q_vecs[ki], shq)
            s_hk = _group_allreduce(_vsum(shk), WTK)
            s_hq = _group_allreduce(_vsum(shq), WTK)
            v_val = fx.Float32(fx.ptr_load(v_lds + gv))
            v_new = (v_val - fx.Float32(s_hk)) * beta
            o_val = fx.Float32(s_hq) + v_new * fx.Float32(dot_kq)
            v_new_vec = fx.Vector.filled(VPT, v_new, fx.Float32)
            upd = []
            for ki in range_constexpr(K_ITERS):
                p = vi * K_ITERS + ki
                upd.append(_fma(k_vecs[ki], v_new_vec, state_vecs[p]))
            if valid:
                for ki in range_constexpr(K_ITERS):
                    kb = k_vec_start + fx.Int32(ki * TILE_K)
                    buf_copy_store(
                        st_buf,
                        gv * fx.Int32(D) + kb,
                        upd[ki],
                        fx.Float32,
                        VPT,
                        cache_modifier=_STATE_CACHE_POLICY,
                    )
            o_bf = valid.select(o_val, fx.Float32(0.0)).to(fx.BFloat16)
            if lane_k == fx.Int32(0):
                out[b_i * stride_out_tok + hv * fx.Int32(D) + gv] = o_bf

        # ---- publish the q/k conv-state write-back flag ------------------------------
        last = ((seen == fx.Int32(RATIO - 1)) & (tid == fx.Int32(0))) & valid
        if tid == fx.Int32(0):
            fx.ptr_store(last.select(fx.Int32(1), fx.Int32(0)), flag_lds)
            if last:
                _ctr_reset(counter_mem, ctr_idx)
        fx.gpu.barrier()
        if (fx.Int32(fx.ptr_load(flag_lds)) == fx.Int32(1)) & valid:
            cs[cqk] = s_qk[1].to(fx.BFloat16)
            cs[cqk + stride_conv_w] = s_qk[2].to(fx.BFloat16)
            cs[cqk + fx.Int32(2) * stride_conv_w] = x_qk.to(fx.BFloat16)

    @flyc.jit
    def launch(
        x_mem: fx.Tensor,
        ba_mem: fx.Tensor,
        conv_w_mem: fx.Tensor,
        conv_state_mem: fx.Tensor,
        counter_mem: fx.Tensor,
        A_log_mem: fx.Tensor,
        dt_bias_mem: fx.Tensor,
        state_mem: fx.Tensor,
        indices_mem: fx.Tensor,
        out_mem: fx.Tensor,
        z_out_mem: fx.Tensor,
        stride_x_tok: fx.Int32,
        stride_ba_tok: fx.Int32,
        stride_conv_slot: fx.Int64,
        stride_conv_ch: fx.Int32,
        stride_conv_w: fx.Int32,
        stride_state_slot: fx.Int64,
        stride_out_tok: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            x_mem,
            ba_mem,
            conv_w_mem,
            conv_state_mem,
            counter_mem,
            A_log_mem,
            dt_bias_mem,
            state_mem,
            indices_mem,
            out_mem,
            z_out_mem,
            stride_x_tok,
            stride_ba_tok,
            stride_conv_slot,
            stride_conv_ch,
            stride_conv_w,
            stride_state_slot,
            stride_out_tok,
        ).launch(
            grid=(batch_size * fx.Int32(HV), 1, 1),
            block=(NT, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": 16,
        }
    }
    return launch


__all__ = ["create_gdn_decode_fused_kernel"]
