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
``B C^T`` with a single ``xor 16`` lane exchange instead of an LDS transpose,
and stages only the strided x tile (plus decay / dt) in LDS.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
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


def _lds_base(storage_type):
    """Byte-addressed base of one SharedAllocator buffer (a plain pointer value, so
    it can cross scf control flow where the storage handle cannot)."""
    lds = fx.SharedAllocator().allocate(storage_type).peek()
    return fx.recast_iter(fx.Uint8, lds.buf.ptr)


def _lds_ptr(base, byte_off, elem):
    ptr_ty = fx.PointerType.get(elem.ir_type, fx.AddressSpace.Shared, elem.width // 8)
    return fx.recast_iter(ptr_ty, fx.add_offset(base, fx.make_int_tuple(byte_off)))


def _lds_load(base, byte_off, elem, n=1):
    v = Vec(
        fx.ptr_load(_lds_ptr(base, byte_off, elem), result_type=Vec.make_type(n, elem))
    )
    return v[0] if n == 1 else v


def _lds_store(base, byte_off, elem, val):
    if not isinstance(val, Vec):
        val = Vec.from_elements([val], elem)
    fx.ptr_store(val, _lds_ptr(base, byte_off, elem))


def _row_view(t):
    """bf16 buffer view addressed by element offset, read 8 elements (128 bits) at a time."""
    return ptr_buf_tensor(t, fx.BFloat16, unit_elems=8, unit_stride=1)


def _load16(view, off):
    """16 contiguous bf16 at element offset ``off`` (16-byte aligned): two dwordx4 loads."""
    lo = buf_copy_load(view, off, fx.BFloat16, unit_elems=8)
    hi = buf_copy_load(view, off + 8, fx.BFloat16, unit_elems=8)
    return lo.shuffle(hi, list(range(16)))


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
    num_heads, num_groups, head_size, state_size, chunk_size, rm, rn, ks
):
    """Carry the state across chunks; one wave owns a (16*rm) x (16*rn) state tile.

    The chunk loop is walked in slabs of ``16 * ks`` tokens (the GEMM K loop of
    ``x^T @ (B * w)``). Each slab's x and B tiles are staged into LDS transposed
    (token-contiguous, so a WMMA operand is two 128-bit LDS reads) with its
    weights ``w[t] = dt[t] * exp2(d_last - d[t])``. The next slab's global loads
    are issued into registers before the current slab's WMMAs and written to LDS
    after them -- the register-staged prefetch of FlyDSL's RDNA3 GEMM, standing
    in for cuTile's load latency hints on this serial loop. ``ks`` trades LDS /
    prefetch registers (occupancy) against barriers per chunk; it is tuned.
    """
    H, G, P, N, L = num_heads, num_groups, head_size, state_size, chunk_size
    HPG = H // G
    SL = TILE * ks  # tokens per slab
    SPC = L // SL  # slabs per chunk
    W_PASSES = (SL + WAVE - 1) // WAVE
    BP, BN = TILE * rm, TILE * rn
    assert P % BP == 0 and N % BN == 0 and L % SL == 0

    # Transposed slab tiles xT[p][t] and bT[n][t]; rows padded by 8 elements.
    XS = SL + 8
    X_OFF = 0
    B_OFF = X_OFF + BP * XS * 2
    W_OFF = B_OFF + BN * XS * 2
    SCALE_OFF = W_OFF + SL * 4
    LDS_TOTAL = SCALE_OFF + 4

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Uint8, LDS_TOTAL, 16]

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
        x_rows = _row_view(x)
        b_rows = _row_view(b)
        dec_r = ptr_buf_tensor(decay, fx.Float32)
        dt_r = ptr_buf_tensor(dt_bht, fx.Float32)
        init_r = ptr_buf_tensor(init_state, fx.Float32)
        hid_w = ptr_buf_tensor(hiddens, fx.BFloat16)
        fin_w = ptr_buf_tensor(final_state, fx.Float32)
        lds = _lds_base(SharedStorage)

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

        tok_base = bh * num_tokens  # [b, h, t] row of decay / dt
        x_row = H * P
        b_row = G * N

        def load_slab(it):
            """Global -> registers for slab ``it``: x / B tiles (128-bit), weights,
            and the state decay to apply before it (1 except on a chunk's first slab).
            """
            c, step = it // SPC, it % SPC
            t0 = c * L + step * SL
            row0 = batch * num_tokens + t0
            xv = []
            for k in range_constexpr(SL * BP // (8 * WAVE)):
                u = lane + k * WAVE
                off = (row0 + u // (BP // 8)) * x_row + h * P + p0 + u % (BP // 8) * 8
                xv.append(buf_copy_load(x_rows, off, fx.BFloat16, unit_elems=8))
            bv = []
            for k in range_constexpr(SL * BN // (8 * WAVE)):
                u = lane + k * WAVE
                off = (row0 + u // (BN // 8)) * b_row + g * N + n0 + u % (BN // 8) * 8
                bv.append(buf_copy_load(b_rows, off, fx.BFloat16, unit_elems=8))
            d_last = fx.Float32(dec_r[tok_base + c * L + L - 1])
            w = []
            for k in range_constexpr(W_PASSES):
                t = (
                    tok_base + t0 + (lane + k * WAVE) % SL
                )  # wrapped lanes are not stored
                w.append(fx.Float32(dt_r[t]) * (d_last - fx.Float32(dec_r[t])).exp2())
            scale = (step == 0).select(d_last.exp2(), fx.Float32(1.0))
            return xv, bv, w, scale

        def store_slab(xv, bv, w, scale):
            """Registers -> LDS, transposing x and B to token-contiguous rows."""
            for k in range_constexpr(len(xv)):
                u = lane + k * WAVE
                s, pc = u // (BP // 8), u % (BP // 8) * 8
                for e in range_constexpr(8):
                    _lds_store(
                        lds, X_OFF + ((pc + e) * XS + s) * 2, fx.BFloat16, xv[k][e]
                    )
            for k in range_constexpr(len(bv)):
                u = lane + k * WAVE
                s, nc = u // (BN // 8), u % (BN // 8) * 8
                for e in range_constexpr(8):
                    _lds_store(
                        lds, B_OFF + ((nc + e) * XS + s) * 2, fx.BFloat16, bv[k][e]
                    )
            for k in range_constexpr(W_PASSES):
                if lane + k * WAVE < SL:
                    _lds_store(lds, W_OFF + (lane + k * WAVE) * 4, fx.Float32, w[k])
            if lane == 0:
                _lds_store(lds, SCALE_OFF, fx.Float32, scale)

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

        num_slabs = num_chunks * SPC
        store_slab(*load_slab(fx.Int32(0)))
        gpu.barrier()

        for it_iv, state in range(0, num_slabs, 1, init=accs):
            it = fx.Int32(it_iv)
            s_accs = _carried(state)
            if it % SPC == 0:
                # Chunk-entry state, narrowed to the dtype compute_out consumes.
                hid_base = (bh * num_chunks + it // SPC) * (P * N)
                for i in range_constexpr(rm):
                    for j in range_constexpr(rn):
                        acc = s_accs[i * rn + j]
                        for si in range_constexpr(8):
                            hid_w[hid_base + state_off(i, j, si)] = fx.Float32(
                                acc[si]
                            ).to(fx.BFloat16)

            # Prefetch the next slab (the last iteration reloads its own, harmlessly).
            nxt = load_slab((it + 1 < num_slabs).select(it + 1, it))

            scale = _lds_load(lds, SCALE_OFF, fx.Float32)
            s_accs = [acc * Vec.filled(8, scale, fx.Float32) for acc in s_accs]
            for kk in range_constexpr(ks):
                t0 = TILE * kk
                wv = _lds_load(lds, W_OFF + t0 * 4, fx.Float32, TILE)
                a_ops = [
                    _lds_load(
                        lds, X_OFF + ((TILE * i + l16) * XS + t0) * 2, fx.BFloat16, TILE
                    )
                    for i in range_constexpr(rm)
                ]
                b_ops = []
                for j in range_constexpr(rn):
                    b_raw = _lds_load(
                        lds, B_OFF + ((TILE * j + l16) * XS + t0) * 2, fx.BFloat16, TILE
                    )
                    b_ops.append(
                        _bf16_vec(
                            [
                                fx.Float32(b_raw[k]) * wv[k]
                                for k in range_constexpr(TILE)
                            ]
                        )
                    )
                for i in range_constexpr(rm):
                    for j in range_constexpr(rn):
                        s_accs[i * rn + j] = _wmma(
                            a_ops[i], b_ops[j], s_accs[i * rn + j]
                        )

            gpu.barrier()
            store_slab(*nxt)
            gpu.barrier()
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
    num_heads, num_groups, head_size, state_size, chunk_size, rp, persistent
):
    """One block per (chunk, p-tile, batch*head); wave w owns token rows 16w..16w+15.

    The block first stages the chunk's x tile (transposed) and its decay / dt
    into LDS with coalesced 128-bit loads, so the waves' source-tile loops read
    the strided x operand from LDS instead of gathering it from memory. B, C and
    the chunk-entry state rows are contiguous and stay 128-bit global reads
    (staging B too costs more occupancy than it saves).

    Tiles are numbered heads-fastest. ``persistent`` launches a fixed grid whose
    blocks stride over the tiles (TileGym's persistent compute_out); otherwise
    each block takes one tile.
    """
    H, G, P, N, L = num_heads, num_groups, head_size, state_size, chunk_size
    HPG = H // G
    NWAVES = L // TILE
    THREADS = NWAVES * WAVE
    NK = N // TILE
    BP = TILE * rp
    NPT = P // BP  # p tiles per head
    assert P % BP == 0 and N % TILE == 0 and L % TILE == 0

    # LDS rows are padded by 8 elements (16 bytes) to spread banks while
    # keeping every 16-element operand read 16-byte aligned.
    XS = L + 8
    X_OFF = 0
    DEC_OFF = X_OFF + BP * XS * 2
    DT_OFF = DEC_OFF + L * 4
    LDS_TOTAL = DT_OFF + L * 4

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Uint8, LDS_TOTAL, 16]

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
        batch_size: Int32,
        num_programs: Int32,
    ):
        x_rows = _row_view(x)
        b_rows = _row_view(b)
        c_rows = _row_view(c)
        hid_rows = _row_view(hiddens)
        d_r = ptr_buf_tensor(d, fx.Float32)
        dec_r = ptr_buf_tensor(decay, fx.Float32)
        dt_r = ptr_buf_tensor(dt_bht, fx.Float32)
        out_w = ptr_buf_tensor(out, fx.BFloat16)
        lds = _lds_base(SharedStorage)

        tid = fx.thread_idx.x
        wave = tid // WAVE
        lane = tid % WAVE
        l16 = lane % 16
        hi = lane // 16
        num_chunks = num_tokens // L

        @flyc.jit
        def process_tile(tile):
            # Heads vary fastest (TileGym _compute_out_tiles): concurrent tiles are
            # the heads of one (batch, chunk), which share its B / C rows in L2.
            h = tile % H
            rest = tile // H
            p0 = rest % NPT * BP
            rest = rest // NPT
            chunk = rest % num_chunks
            batch = rest // num_chunks
            bh = batch * H + h
            g = h // HPG

            t_chunk = chunk * L
            tok_base = bh * num_tokens + t_chunk  # into decay / dt [b, h, t]
            x_row = H * P
            bc_row = G * N
            bc_chunk = (batch * num_tokens + t_chunk) * bc_row + g * N
            x_chunk = (batch * num_tokens + t_chunk) * x_row + h * P

            # ---- stage the chunk into LDS ----
            for k in range_constexpr(BP // 16):  # x tile, written transposed: xT[p][s]
                u = tid + k * THREADS
                s = u // (BP // 8)
                pc = u % (BP // 8) * 8
                v = buf_copy_load(
                    x_rows, x_chunk + s * x_row + p0 + pc, fx.BFloat16, unit_elems=8
                )
                for j in range_constexpr(8):
                    _lds_store(lds, X_OFF + ((pc + j) * XS + s) * 2, fx.BFloat16, v[j])
            if tid < L:
                _lds_store(
                    lds,
                    DEC_OFF + tid * 4,
                    fx.Float32,
                    fx.Float32(dec_r[tok_base + tid]),
                )
                _lds_store(
                    lds, DT_OFF + tid * 4, fx.Float32, fx.Float32(dt_r[tok_base + tid])
                )
            gpu.barrier()

            def c_op(kn):
                return _load16(c_rows, c_base + TILE * kn)

            # A-operand row of this lane: token t_a = 16*wave + l16 of the chunk.
            t_a = TILE * wave + l16
            d_ta = _lds_load(lds, DEC_OFF + t_a * 4, fx.Float32)
            c_base = bc_chunk + t_a * bc_row

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
                    h_op = _load16(
                        hid_rows, hid_base + (p0 + TILE * i + l16) * N + TILE * kn
                    )
                    accs[i] = _wmma(a_op, h_op, accs[i])

            # Intra-chunk term over source tiles s <= t (causal, so tiles 0..wave).
            for st_iv, state in range(0, wave + 1, 1, init=accs):
                s_accs = _carried(state)
                s0 = fx.Int32(st_iv) * TILE
                # S^T[s, t] = B[s] . C[t]: A = B rows (lane row s0 + l16), B = C rows (lane col t_a).
                sT = Vec.filled(8, 0.0, fx.Float32)
                b_base = bc_chunk + (s0 + l16) * bc_row
                for kn in range_constexpr(NK):
                    sT = _wmma(_load16(b_rows, b_base + TILE * kn), c_op(kn), sT)
                # Slot si holds source s = s0 + 2*si + hi for target t_a.
                own = []
                for si in range_constexpr(8):
                    s = s0 + 2 * si + hi
                    d_s = _lds_load(lds, DEC_OFF + s * 4, fx.Float32)
                    dt_s = _lds_load(lds, DT_OFF + s * 4, fx.Float32)
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
                    x_op = _lds_load(
                        lds, X_OFF + ((TILE * i + l16) * XS + s0) * 2, fx.BFloat16, TILE
                    )
                    s_accs[i] = _wmma(p_op, x_op, s_accs[i])
                results = yield s_accs

            d_h = fx.Float32(d_r[h])
            for i in range_constexpr(rp):
                acc = _carried(results)[i]
                for si in range_constexpr(8):
                    t = TILE * wave + 2 * si + hi
                    x_t = _lds_load(
                        lds, X_OFF + ((TILE * i + l16) * XS + t) * 2, fx.BFloat16
                    )
                    y = fx.Float32(acc[si]) + d_h * fx.Float32(x_t)
                    out_w[x_chunk + t * x_row + p0 + TILE * i + l16] = y.to(fx.BFloat16)

        if const_expr(persistent):
            num_tiles = batch_size * num_chunks * NPT * H
            for tile in range(fx.block_idx.x, num_tiles, num_programs):
                process_tile(fx.Int32(tile))
                gpu.barrier()  # the next tile restages LDS
        else:
            process_tile(fx.Int32(fx.block_idx.x))

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
        batch: Int32,
        num_programs: Int32,
        stream: fx.Stream,
    ):
        compute_out_kernel(
            x, b, c, d, decay, dt_bht, hiddens, out, num_tokens, batch, num_programs
        ).launch(
            grid=(fx.Int64(num_programs), 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch
