# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL Mamba-2 SSD chunked forward kernels (gfx11 WMMA, wave32).

Port of TileGym's cuTile ``mamba2_chunk_forward``, split into the same three
kernels:

* ``chunk_cumsum``  -- per-chunk cumulative log2 decay ``LOG2E * cumsum(a*dt)``,
  written token-contiguous (``[b, h, t]``) so later kernels read it with
  wave-uniform loads, plus ``dt`` in the same layout.
* ``fwd_pass_state`` -- sequential walk over chunks carrying the
  ``[head_size, state_size]`` state in WMMA accumulators; stores each chunk's
  entry state (narrowed, the operand ``compute_out`` consumes) and the fp32
  final state.
* ``compute_out``   -- fuses the carried-state term ``exp2(d_t) * C h^T``, the
  masked intra-chunk term ``(C B^T * L) (dt x)`` and the ``D x`` skip without
  materialising the ``[chunk, chunk]`` interaction matrix.

gfx11 WMMA ``16x16x16`` (v16 operands, v8 f32 accumulator) lane contract:
  A operand : lane L carries row ``L % 16``, K = 0..15
  B operand : lane L carries column ``L % 16``, K = 0..15
  C/D       : lane L, slot si holds ``D[2*si + L // 16][L % 16]``
Operands are chosen so every load is either contiguous per lane or coalesced
across lanes 0-15; ``compute_out`` builds its P operand from an accumulator of
``B C^T`` with a single ``xor 16`` lane exchange, so no LDS round trip is needed.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_load, ptr_buf_tensor

WAVE = 32
TILE = 16  # WMMA M = N = K


def _wmma(a_v, b_v, acc):
    """``acc += A @ B`` with bf16 v16 operands and a v8 f32 accumulator."""
    return Vec(
        rocdl.wmma_f32_16x16x16_bf16(
            acc.type, a_v.bitcast(fx.Int16), b_v.bitcast(fx.Int16), acc
        ).result
    )


def _bf16_vec(vals):
    return Vec.from_elements([fx.Float32(v).to(fx.BFloat16) for v in vals], fx.BFloat16)


def _carried(state):
    """Loop-carried accumulators as a list; a lone loop result comes back bare."""
    return [Vec(v) for v in state] if isinstance(state, (list, tuple)) else [Vec(state)]


def _row_view(t, vec_loads):
    """bf16 buffer view addressed by element offset; 8-wide (128-bit) units if ``vec_loads``."""
    if vec_loads:
        return ptr_buf_tensor(t, fx.BFloat16, unit_elems=8, unit_stride=1)
    return ptr_buf_tensor(t, fx.BFloat16)


def _load16(view, off, vec_loads):
    """16 contiguous bf16 at element offset ``off``: two dwordx4 loads or 16 scalar ones."""
    if vec_loads:
        lo = buf_copy_load(view, off, fx.BFloat16, unit_elems=8)
        hi = buf_copy_load(view, off + 8, fx.BFloat16, unit_elems=8)
        return lo.shuffle(hi, list(range(16)))
    return Vec.from_elements(
        [view[off + k] for k in range_constexpr(TILE)], fx.BFloat16
    )


def build_chunk_cumsum(num_heads, chunk_size, block_h):
    """dt[b, t, h] -> decay[b, h, t] (f32, log2 units) and dt_bht[b, h, t] (f32)."""
    H, L, BH = num_heads, chunk_size, block_h

    @flyc.kernel
    def chunk_cumsum_kernel(
        a: fx.Tensor,
        dt: fx.Tensor,
        decay: fx.Tensor,
        dt_bht: fx.Tensor,
        num_tokens: Int32,
    ):
        a_r = ptr_buf_tensor(a, fx.Float32)
        dt_r = ptr_buf_tensor(dt, fx.BFloat16)
        dec_w = ptr_buf_tensor(decay, fx.Float32)
        dtw = ptr_buf_tensor(dt_bht, fx.Float32)

        chunk = fx.block_idx.x
        h = fx.block_idx.y * BH + fx.thread_idx.x
        batch = fx.block_idx.z
        if h < H:
            a_h = fx.Float32(a_r[h])
            t0 = chunk * L
            src = (batch * num_tokens + t0) * H + h
            dst = (batch * H + h) * num_tokens + t0
            acc = fx.Float32(0.0)
            for k in range_constexpr(L):
                dtk = fx.Float32(dt_r[src + k * H])
                acc = acc + a_h * dtk
                dec_w[dst + k] = acc * LOG2E
                dtw[dst + k] = dtk

    @flyc.jit
    def launch(
        a: fx.Tensor,
        dt: fx.Tensor,
        decay: fx.Tensor,
        dt_bht: fx.Tensor,
        num_tokens: Int32,
        num_chunks: Int32,
        batch: Int32,
        stream: fx.Stream,
    ):
        chunk_cumsum_kernel(a, dt, decay, dt_bht, num_tokens).launch(
            grid=(fx.Int64(num_chunks), (H + BH - 1) // BH, fx.Int64(batch)),
            block=(BH, 1, 1),
            stream=stream,
        )

    return launch


def build_fwd_pass_state(
    num_heads, num_groups, head_size, state_size, chunk_size, rm, rn
):
    """Carry the state across chunks; one wave owns a (16*rm) x (16*rn) state tile."""
    H, G, P, N, L = num_heads, num_groups, head_size, state_size, chunk_size
    HPG = H // G
    KT = L // TILE
    BP, BN = TILE * rm, TILE * rn
    assert P % BP == 0 and N % BN == 0 and L % TILE == 0

    @flyc.kernel
    def fwd_pass_state_kernel(
        x: fx.Tensor,
        b: fx.Tensor,
        decay: fx.Tensor,
        dt_bht: fx.Tensor,
        init_state: fx.Tensor,
        hiddens: fx.Tensor,
        final_state: fx.Tensor,
        num_tokens: Int32,
    ):
        x_r = ptr_buf_tensor(x, fx.BFloat16)
        b_r = ptr_buf_tensor(b, fx.BFloat16)
        dec_r = ptr_buf_tensor(decay, fx.Float32)
        dt_r = ptr_buf_tensor(dt_bht, fx.Float32)
        init_r = ptr_buf_tensor(init_state, fx.Float32)
        hid_w = ptr_buf_tensor(hiddens, fx.BFloat16)
        fin_w = ptr_buf_tensor(final_state, fx.Float32)

        lane = fx.thread_idx.x
        l16 = lane % 16
        hi = lane // 16
        p0 = fx.block_idx.x * BP
        n0 = fx.block_idx.y * BN
        bh = fx.block_idx.z
        batch = bh // H
        h = bh % H
        g = h // HPG
        num_chunks = num_tokens // L

        # Accumulator element (rm, rn, si) is state[p0 + 16*rm + 2*si + hi, n0 + 16*rn + l16].
        def state_off(i, j, si):
            return (p0 + TILE * i + 2 * si + hi) * N + n0 + TILE * j + l16

        state_base = bh * (P * N)
        accs = []
        for i in range_constexpr(rm):
            for j in range_constexpr(rn):
                accs.append(
                    Vec.from_elements(
                        [
                            fx.Float32(init_r[state_base + state_off(i, j, si)])
                            for si in range_constexpr(8)
                        ],
                        fx.Float32,
                    )
                )

        tok_base = bh * num_tokens  # [b, h, t] row of decay / dt
        x_row = H * P
        b_row = G * N

        for c_iv, state in range(0, num_chunks, 1, init=accs):
            c = fx.Int32(c_iv)
            s_accs = _carried(state)
            # Chunk-entry state, narrowed to the dtype compute_out consumes.
            hid_base = (bh * num_chunks + c) * (P * N)
            for i in range_constexpr(rm):
                for j in range_constexpr(rn):
                    acc = s_accs[i * rn + j]
                    for si in range_constexpr(8):
                        hid_w[hid_base + state_off(i, j, si)] = fx.Float32(acc[si]).to(
                            fx.BFloat16
                        )

            t_chunk = c * L
            d_last = fx.Float32(dec_r[tok_base + t_chunk + L - 1])
            scale = d_last.exp2()
            s_accs = [acc * Vec.filled(8, scale, fx.Float32) for acc in s_accs]

            for kk in range_constexpr(KT):
                t0 = t_chunk + TILE * kk
                # w[t] = dt[t] * exp2(d_last - d[t]); wave-uniform.
                w = [
                    fx.Float32(dt_r[tok_base + t0 + k])
                    * (d_last - fx.Float32(dec_r[tok_base + t0 + k])).exp2()
                    for k in range_constexpr(TILE)
                ]
                x_tok = (batch * num_tokens + t0) * x_row + h * P + p0 + l16
                b_tok = (batch * num_tokens + t0) * b_row + g * N + n0 + l16
                a_ops = [
                    Vec.from_elements(
                        [
                            x_r[x_tok + k * x_row + TILE * i]
                            for k in range_constexpr(TILE)
                        ],
                        fx.BFloat16,
                    )
                    for i in range_constexpr(rm)
                ]
                b_ops = [
                    _bf16_vec(
                        [
                            fx.Float32(b_r[b_tok + k * b_row + TILE * j]) * w[k]
                            for k in range_constexpr(TILE)
                        ]
                    )
                    for j in range_constexpr(rn)
                ]
                for i in range_constexpr(rm):
                    for j in range_constexpr(rn):
                        s_accs[i * rn + j] = _wmma(
                            a_ops[i], b_ops[j], s_accs[i * rn + j]
                        )

            results = yield s_accs

        for i in range_constexpr(rm):
            for j in range_constexpr(rn):
                acc = _carried(results)[i * rn + j]
                for si in range_constexpr(8):
                    fin_w[state_base + state_off(i, j, si)] = fx.Float32(acc[si])

    @flyc.jit
    def launch(
        x: fx.Tensor,
        b: fx.Tensor,
        decay: fx.Tensor,
        dt_bht: fx.Tensor,
        init_state: fx.Tensor,
        hiddens: fx.Tensor,
        final_state: fx.Tensor,
        num_tokens: Int32,
        batch: Int32,
        stream: fx.Stream,
    ):
        fwd_pass_state_kernel(
            x, b, decay, dt_bht, init_state, hiddens, final_state, num_tokens
        ).launch(
            grid=(P // BP, N // BN, fx.Int64(batch * H)),
            block=(WAVE, 1, 1),
            stream=stream,
        )

    return launch


def build_compute_out(
    num_heads, num_groups, head_size, state_size, chunk_size, rp, vec_loads
):
    """One block per (chunk, p-tile, batch*head); wave w owns token rows 16w..16w+15.

    ``vec_loads`` reads the contiguous B / C / state rows 128 bits at a time;
    which width wins depends on the shape, so it is tuned.
    """
    H, G, P, N, L = num_heads, num_groups, head_size, state_size, chunk_size
    HPG = H // G
    NWAVES = L // TILE
    NK = N // TILE
    BP = TILE * rp
    assert P % BP == 0 and N % TILE == 0 and L % TILE == 0

    @flyc.kernel
    def compute_out_kernel(
        x: fx.Tensor,
        b: fx.Tensor,
        c: fx.Tensor,
        d: fx.Tensor,
        decay: fx.Tensor,
        dt_bht: fx.Tensor,
        hiddens: fx.Tensor,
        out: fx.Tensor,
        num_tokens: Int32,
    ):
        x_r = ptr_buf_tensor(x, fx.BFloat16)
        b_rows = _row_view(b, vec_loads)
        c_rows = _row_view(c, vec_loads)
        d_r = ptr_buf_tensor(d, fx.Float32)
        dec_r = ptr_buf_tensor(decay, fx.Float32)
        dt_r = ptr_buf_tensor(dt_bht, fx.Float32)
        hid_rows = _row_view(hiddens, vec_loads)
        out_w = ptr_buf_tensor(out, fx.BFloat16)

        tid = fx.thread_idx.x
        wave = tid // WAVE
        lane = tid % WAVE
        l16 = lane % 16
        hi = lane // 16
        chunk = fx.block_idx.x
        p0 = fx.block_idx.y * BP
        bh = fx.block_idx.z
        batch = bh // H
        h = bh % H
        g = h // HPG
        num_chunks = num_tokens // L

        t_chunk = chunk * L
        tok_base = bh * num_tokens + t_chunk  # into decay / dt [b, h, t]
        x_row = H * P
        bc_row = G * N
        bc_chunk = (batch * num_tokens + t_chunk) * bc_row + g * N
        x_chunk = (batch * num_tokens + t_chunk) * x_row + h * P

        # A-operand row of this lane: token t_a = 16*wave + l16 of the chunk.
        t_a = TILE * wave + l16
        d_ta = fx.Float32(dec_r[tok_base + t_a])

        c_base = bc_chunk + t_a * bc_row

        def c_op(kn):
            return _load16(c_rows, c_base + TILE * kn, vec_loads)

        # Carried-state term: (exp2(d_t) * C) @ h^T, h = chunk-entry state [P, N].
        hid_base = (bh * num_chunks + chunk) * (P * N)
        exp_dt = d_ta.exp2()
        accs = [Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(rp)]
        for kn in range_constexpr(NK):
            c_kn = c_op(kn)
            a_op = _bf16_vec(
                [fx.Float32(c_kn[k]) * exp_dt for k in range_constexpr(TILE)]
            )
            for i in range_constexpr(rp):
                b_op = _load16(
                    hid_rows,
                    hid_base + (p0 + TILE * i + l16) * N + TILE * kn,
                    vec_loads,
                )
                accs[i] = _wmma(a_op, b_op, accs[i])

        # Intra-chunk term over source tiles s <= t (causal, so tiles 0..wave).
        for st_iv, state in range(0, wave + 1, 1, init=accs):
            st = fx.Int32(st_iv)
            s_accs = _carried(state)
            s0 = st * TILE
            # S^T[s, t] = B[s] . C[t]: A = B rows (lane row s0 + l16), B = C rows (lane col t_a).
            sT = Vec.filled(8, 0.0, fx.Float32)
            b_base = bc_chunk + (s0 + l16) * bc_row
            for kn in range_constexpr(NK):
                sT = _wmma(_load16(b_rows, b_base + TILE * kn, vec_loads), c_op(kn), sT)
            # Slot si holds source s = s0 + 2*si + hi for target t_a.
            own = []
            for si in range_constexpr(8):
                s = s0 + 2 * si + hi
                d_s = fx.Float32(dec_r[tok_base + s])
                dt_s = fx.Float32(dt_r[tok_base + s])
                expo = (s <= t_a).select(d_ta - d_s, fx.Float32(float("-inf")))
                own.append(fx.Float32(sT[si]) * expo.exp2() * dt_s)
            other = [v.shuffle_xor(16, WAVE) for v in own]
            # P[t_a, s0 + k]: even k from the hi == 0 half, odd k from hi == 1.
            is_lo = hi == 0
            p_vals = []
            for si in range_constexpr(8):
                p_vals.append(is_lo.select(own[si], other[si]))
                p_vals.append(is_lo.select(other[si], own[si]))
            p_op = _bf16_vec(p_vals)
            for i in range_constexpr(rp):
                xb = x_chunk + s0 * x_row + p0 + TILE * i + l16
                x_op = Vec.from_elements(
                    [x_r[xb + k * x_row] for k in range_constexpr(TILE)], fx.BFloat16
                )
                s_accs[i] = _wmma(p_op, x_op, s_accs[i])
            results = yield s_accs

        d_h = fx.Float32(d_r[h])
        for i in range_constexpr(rp):
            acc = _carried(results)[i]
            for si in range_constexpr(8):
                t = TILE * wave + 2 * si + hi
                off = x_chunk + t * x_row + p0 + TILE * i + l16
                y = fx.Float32(acc[si]) + d_h * fx.Float32(x_r[off])
                out_w[off] = y.to(fx.BFloat16)

    @flyc.jit
    def launch(
        x: fx.Tensor,
        b: fx.Tensor,
        c: fx.Tensor,
        d: fx.Tensor,
        decay: fx.Tensor,
        dt_bht: fx.Tensor,
        hiddens: fx.Tensor,
        out: fx.Tensor,
        num_tokens: Int32,
        num_chunks: Int32,
        batch: Int32,
        stream: fx.Stream,
    ):
        compute_out_kernel(x, b, c, d, decay, dt_bht, hiddens, out, num_tokens).launch(
            grid=(fx.Int64(num_chunks), P // BP, fx.Int64(batch * H)),
            block=(NWAVES * WAVE, 1, 1),
            stream=stream,
        )

    return launch
