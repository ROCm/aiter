# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated DeltaNet prefill front end for gfx942: in_proj qkvz rows -> varlen causal
conv (width 4) + SiLU -> q/k/v split -> q/k L2 norm, in one pass.

Grid (2H + HV, max_chunks, num_seqs): one workgroup per (head, 128-token chunk,
sequence). Heads vary fastest, so concurrent workgroups read neighbouring 256-byte
head slices of the same rows; each still reads its rows at the qkvz row stride. With
8 KiB rows (TP4) the time depends on the qkvz / output placement: the wrapper skips it.
Each thread owns 8 channels (one 16-byte vector) of 8 consecutive tokens and keeps
the 3-token causal halo in registers. The 16 lanes of a token sit in one DPP row,
so the per-token L2 norm is a DPP all-reduce without LDS.

Numerics follow the Triton kernels: each product is rounded to bf16 and accumulated
in fp32, SiLU, rounded to bf16, then y / sqrt(sum(y^2) + eps) for q and k. The
chunk-0 workgroup of a sequence reads the initial conv state (when has_initial_state)
and writes the final conv state (the raw inputs of the last 3 tokens).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

from aiter.ops.flydsl.kernels.act import sigmoid_f32
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from aiter.ops.flydsl.kernels.tensor_shim import (
    buf_base_i64,
    buf_copy_load,
    buf_copy_store,
    buf_scalar_load,
    ptr_buf_tensor,
)

_D = 128
_OCT = 8  # channels per thread: one 16-byte vector
_LANES_PER_TOK = _D // _OCT  # 16, one DPP row
_NT = 256
_TG = 8  # tokens per thread
CHUNK_TOKENS = (_NT // _LANES_PER_TOK) * _TG  # 128


def _row16_sum(x):
    """Sum over the 16 lanes of a DPP row: xor1, xor2, half_mirror, mirror."""
    for ctrl in (177, 78, 0x141, 0x140):
        i = fx.Float32(x).bitcast(fx.Int32)
        x = fx.Float32(x) + fx.Int32(update_dpp_i32(i, i, ctrl)).bitcast(fx.Float32)
    return x


def _sload_i32(mem, idx):
    """Uniform int32 element through the scalar unit."""
    return fx.Int32(buf_scalar_load(ptr_buf_tensor(mem, fx.Int32), idx))


def _bf16_round(x):
    """Round to nearest even to bf16, kept in f32 (finite inputs)."""
    b = fx.Float32(x).bitcast(fx.Int32)
    lsb = (b >> fx.Int32(16)) & fx.Int32(1)
    return ((b + fx.Int32(0x7FFF) + lsb) & fx.Int32(-65536)).bitcast(fx.Float32)


def _bf16_round_product(p):
    """RNE to bf16 of an exact bf16 x bf16 product by Veltkamp splitting. Pure FP,
    so it packs into v_pk_mul/v_pk_add_f32; bit-identical to RNE for such products."""
    c = fx.Float32(p) * fx.Float32(65537.0)
    return c - (c - fx.Float32(p))


def _pack8(vals):
    """8 bf16-exact f32 values -> 4 x i32 (the high halves, pairwise)."""
    words = []
    for c in range_constexpr(_OCT // 2):
        lo = (vals[2 * c].bitcast(fx.Int32) >> fx.Int32(16)) & fx.Int32(0xFFFF)
        hi = vals[2 * c + 1].bitcast(fx.Int32) & fx.Int32(-65536)
        words.append(hi | lo)
    return fx.Vector.from_elements(words, fx.Int32)


def _load8(view, unit_idx):
    """8 bf16 values at a 16-byte unit index, as a list of f32."""
    v = fx.Vector(buf_copy_load(view, unit_idx, fx.BFloat16, _OCT).to(fx.Float32))
    return [v[i] for i in range(_OCT)]


@functools.cache
def create_gdn_prefill_front_kernel(
    num_k_heads: int, num_v_heads: int, l2_eps: float = 1e-6
):
    H, HV, D = num_k_heads, num_v_heads, _D
    QK_DIM = 2 * H * D
    NG = 2 * H + HV
    name = f"gdn_prefill_front_h{H}x{HV}_d{D}"

    @flyc.kernel(name=name, known_block_size=[_NT, 1, 1])
    def kernel(
        x_mem: fx.Tensor,  # qkvz [T, >= QK_DIM + HV*D] bf16, rows 16-byte aligned
        conv_w_mem: fx.Tensor,  # [conv_dim, 4] bf16, contiguous
        conv_state_mem: fx.Tensor,  # [slots, conv_dim, 3] bf16, slot rows contiguous
        cache_idx_mem: fx.Tensor,  # int32 [N]
        has_init_mem: fx.Tensor,  # int32 [N]
        cu_mem: fx.Tensor,  # int32 [N + 1]
        q_mem: fx.Tensor,  # bf16 [T, H, D]
        k_mem: fx.Tensor,  # bf16 [T, H, D]
        v_mem: fx.Tensor,  # bf16 [T, HV, D]
        x_row_units: fx.Int32,  # qkvz row stride in 16-byte units
        conv_slot_bytes: fx.Int64,  # conv state slot stride in bytes
    ):
        # 16-byte views: one unit = 8 bf16 = 4 i32
        x = ptr_buf_tensor(x_mem, fx.BFloat16, unit_elems=_OCT)
        conv_w = ptr_buf_tensor(conv_w_mem, fx.BFloat16, unit_elems=_OCT)
        out_q = ptr_buf_tensor(q_mem, fx.Int32, unit_elems=4)
        out_k = ptr_buf_tensor(k_mem, fx.Int32, unit_elems=4)
        out_v = ptr_buf_tensor(v_mem, fx.Int32, unit_elems=4)

        tid = fx.thread_idx.x
        grp = fx.block_idx.x
        chunk = fx.block_idx.y
        seq = fx.block_idx.z
        start = _sload_i32(cu_mem, seq)
        seqlen = _sload_i32(cu_mem, seq + fx.Int32(1)) - start
        slot = _sload_i32(cache_idx_mem, seq)
        has_init = _sload_i32(has_init_mem, seq) != fx.Int32(0)

        oct_ = tid % fx.Int32(_LANES_PER_TOK)
        tgi = tid // fx.Int32(_LANES_PER_TOK)
        tok0 = chunk * fx.Int32(CHUNK_TOKENS) + tgi * fx.Int32(_TG)

        is_q = grp < fx.Int32(H)
        is_k = (grp >= fx.Int32(H)) & (grp < fx.Int32(2 * H))
        is_qk = grp < fx.Int32(2 * H)
        is_v = grp >= fx.Int32(2 * H)
        ch0 = is_qk.select(
            grp * fx.Int32(D), fx.Int32(QK_DIM) + (grp - fx.Int32(2 * H)) * fx.Int32(D)
        )
        ch_u = ch0 // fx.Int32(_OCT) + oct_  # my 8 channels, in 16-byte units

        active = (chunk * fx.Int32(CHUNK_TOKENS) < seqlen) & (slot >= fx.Int32(0))
        if active:
            # conv taps of my 8 channels: w[c*4 + j], 32 values in 4 units
            wv = []
            for i in range_constexpr(4):
                wv += _load8(conv_w, ch_u * fx.Int32(4) + fx.Int32(i))
            # initial conv state of my 8 channels: s[c*3 + k], 24 values in 3 units
            cs = ptr_buf_tensor(
                buf_base_i64(conv_state_mem) + fx.Int64(slot) * conv_slot_bytes,
                fx.BFloat16,
                unit_elems=_OCT,
            )
            sv = []
            for i in range_constexpr(3):
                sv += _load8(cs, ch_u * fx.Int32(3) + fx.Int32(i))

            def init_value(c, ri):
                # rows before the sequence (ri in -3..-1) come from conv state k = 3 + ri
                s = (ri == fx.Int32(-3)).select(
                    sv[c * 3], (ri == fx.Int32(-2)).select(sv[c * 3 + 1], sv[c * 3 + 2])
                )
                return has_init.select(s, fx.Float32(0.0))

            # raw conv input rows tok0-3 .. tok0+TG-1, clamped into the sequence
            rows = []
            last = seqlen - fx.Int32(1)
            for r in range_constexpr(_TG + 3):
                ri = tok0 + fx.Int32(r - 3)
                rc = (ri < fx.Int32(0)).select(
                    fx.Int32(0), (ri > last).select(last, ri)
                )
                vals = _load8(x, (start + rc) * x_row_units + ch_u)
                if fx.const_expr(r < 3):
                    before = ri < fx.Int32(0)
                    vals = [
                        before.select(init_value(c, ri), vals[c]) for c in range(_OCT)
                    ]
                rows.append(vals)

            for t in range_constexpr(_TG):
                ys = []
                for c in range_constexpr(_OCT):
                    acc = fx.Float32(0.0)
                    for j in range_constexpr(4):
                        acc = acc + _bf16_round_product(rows[t + j][c] * wv[c * 4 + j])
                    ys.append(_bf16_round(acc * sigmoid_f32(acc)))
                ovec = _pack8(ys)
                # q/k heads only (uniform per workgroup): v heads skip the norm
                if is_qk:
                    ss = fx.Float32(0.0)
                    for c in range_constexpr(_OCT):
                        ss = ss + ys[c] * ys[c]
                    ss = _row16_sum(ss)
                    inv = fx.Float32(1.0) / fx.math.sqrt(ss + fx.Float32(l2_eps))
                    ovec = _pack8([_bf16_round(ys[c] * inv) for c in range(_OCT)])
                tok = tok0 + fx.Int32(t)
                gtok = start + tok
                in_seq = tok < seqlen
                qk_u = (
                    gtok * fx.Int32(H * D // _OCT)
                    + (grp % fx.Int32(H)) * fx.Int32(D // _OCT)
                    + oct_
                )
                v_u = (
                    gtok * fx.Int32(HV * D // _OCT)
                    + (grp - fx.Int32(2 * H)) * fx.Int32(D // _OCT)
                    + oct_
                )
                if in_seq & is_q:
                    buf_copy_store(out_q, qk_u, ovec, fx.Int32, 4)
                if in_seq & is_k:
                    buf_copy_store(out_k, qk_u, ovec, fx.Int32, 4)
                if in_seq & is_v:
                    buf_copy_store(out_v, v_u, ovec, fx.Int32, 4)

            # Final conv state, by the chunk-0 / token-group-0 threads. They read the
            # initial state above, so the in-place update needs no barrier.
            if (chunk == fx.Int32(0)) & (tgi == fx.Int32(0)):
                news = []
                tail = []
                for k3 in range_constexpr(3):
                    ri = seqlen - fx.Int32(3 - k3)
                    rc = (ri < fx.Int32(0)).select(fx.Int32(0), ri)
                    tail.append((ri, _load8(x, (start + rc) * x_row_units + ch_u)))
                for c in range_constexpr(_OCT):
                    for k3 in range_constexpr(3):
                        ri, vals = tail[k3]
                        news.append(
                            (ri < fx.Int32(0)).select(init_value(c, ri), vals[c])
                        )
                cs_out = ptr_buf_tensor(
                    buf_base_i64(conv_state_mem) + fx.Int64(slot) * conv_slot_bytes,
                    fx.Int32,
                    unit_elems=4,
                )
                for i in range_constexpr(3):
                    buf_copy_store(
                        cs_out,
                        ch_u * fx.Int32(3) + fx.Int32(i),
                        _pack8(news[i * _OCT : (i + 1) * _OCT]),
                        fx.Int32,
                        4,
                    )

    @flyc.jit
    def launch(
        x_mem: fx.Tensor,
        conv_w_mem: fx.Tensor,
        conv_state_mem: fx.Tensor,
        cache_idx_mem: fx.Tensor,
        has_init_mem: fx.Tensor,
        cu_mem: fx.Tensor,
        q_mem: fx.Tensor,
        k_mem: fx.Tensor,
        v_mem: fx.Tensor,
        x_row_units: fx.Int32,
        conv_slot_bytes: fx.Int64,
        max_chunks: fx.Int32,
        num_seqs: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            x_mem,
            conv_w_mem,
            conv_state_mem,
            cache_idx_mem,
            has_init_mem,
            cu_mem,
            q_mem,
            k_mem,
            v_mem,
            x_row_units,
            conv_slot_bytes,
        ).launch(grid=(NG, max_chunks, num_seqs), block=(_NT, 1, 1), stream=stream)

    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": 16,
        }
    }
    return launch


__all__ = ["CHUNK_TOKENS", "create_gdn_prefill_front_kernel"]
