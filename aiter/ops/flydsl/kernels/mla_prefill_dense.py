# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dense absorbed MLA prefill (FlyDSL) — Kimi / SILOTIGER-957 context path.

Contract (absorb / latent, no head padding):
  Q  : bf16 [total_q, H, 576]
  KV : bf16 flat rows of 576; V is the leading 512
  O  : bf16 [total_q, H, 512]
  Mask: ``is_causal`` compile-time (default False)

Schedule:
  Grid (num_q_tiles, H). CTA = 4 waves / 256 threads.
  Each warp owns one Q row of a BLOCK_M=4 tile. Host pads each sequence so
  every tile is full (all warps compute — no producer-only waves).
  KV staged to LDS in BLOCK_N-row tiles and reused across the Q rows in the CTA.

NOTE: Do NOT use ``from __future__ import annotations``.
"""

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath

QK_HEAD_DIM = 576
V_HEAD_DIM = 512
WARP_SIZE = 64
NUM_WARPS = 4
NUM_THREADS = NUM_WARPS * WARP_SIZE
BLOCK_M = NUM_WARPS
VEC_QK = QK_HEAD_DIM // WARP_SIZE  # 9
VEC_V = V_HEAD_DIM // WARP_SIZE  # 8

DEFAULT_BLOCK_N = 32
DEFAULT_WAVES_PER_EU = 1
_LOG2E = math.log2(math.e)
_NEG_INF = -3.4028234663852886e38
SUPPORTED_GFX = ("gfx942", "gfx950")


@lru_cache(maxsize=32)
def compile_mla_prefill_dense(
    *,
    block_n: int = DEFAULT_BLOCK_N,
    is_causal: bool = False,
    waves_per_eu: int = DEFAULT_WAVES_PER_EU,
):
    if block_n <= 0 or (block_n & (block_n - 1)) != 0:
        raise ValueError(f"block_n must be a power of 2, got {block_n}")
    if block_n * QK_HEAD_DIM * 2 > 64 * 1024:
        raise ValueError(f"block_n={block_n} exceeds gfx942 LDS budget")
    BLOCK_N = int(block_n)
    CAUSAL = bool(is_causal)
    LDS_KV_ELEMS = BLOCK_N * QK_HEAD_DIM
    ELEMS_PER_THR = LDS_KV_ELEMS // NUM_THREADS

    kernel_value_attrs = (
        {"rocdl.waves_per_eu": int(waves_per_eu)} if waves_per_eu >= 1 else {}
    )
    kernel_compile_hints = (
        {"waves_per_eu": int(waves_per_eu)} if waves_per_eu >= 1 else {}
    )

    @fx.struct
    class SharedStorage:
        kv: fx.Array[fx.BFloat16, LDS_KV_ELEMS, 16]

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def kn_mla_prefill_dense(
        q: fx.Tensor,
        kv: fx.Tensor,
        o: fx.Tensor,
        tile_q_start: fx.Tensor,
        tile_batch: fx.Tensor,
        qo_indptr: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_indices: fx.Tensor,
        page_size: fx.Int32,
        num_heads: fx.Int32,
        num_tiles: fx.Int32,
        sm_scale: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        tile = fx.Int32(gpu.block_id("x"))
        head = fx.Int32(gpu.block_id("y"))
        warp = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)

        q_buf = fx.rocdl.make_buffer_tensor(q)
        kv_buf = fx.rocdl.make_buffer_tensor(kv)
        o_buf = fx.rocdl.make_buffer_tensor(o)
        tqs_buf = fx.rocdl.make_buffer_tensor(tile_q_start)
        tb_buf = fx.rocdl.make_buffer_tensor(tile_batch)
        qo_buf = fx.rocdl.make_buffer_tensor(qo_indptr)
        kv_ip_buf = fx.rocdl.make_buffer_tensor(kv_indptr)
        kv_ix_buf = fx.rocdl.make_buffer_tensor(kv_indices)
        lds_kv = fx.SharedAllocator().allocate(SharedStorage).peek().kv

        active = (tile < num_tiles) & (head < num_heads)

        def _body():
            q0 = fx.Int32(tqs_buf[tile])
            b = fx.Int32(tb_buf[tile])
            q_seq0 = fx.Int32(qo_buf[b])
            q_seq1 = fx.Int32(qo_buf[b + 1])
            q_len = q_seq1 - q_seq0
            kv_start = fx.Int32(kv_ip_buf[b])
            kv_end = fx.Int32(kv_ip_buf[b + 1])
            kv_len = kv_end - kv_start

            # Full tiles only (host pads). Every warp owns a live Q row.
            q_row = q0 + warp
            q_pos = q_row - q_seq0

            qk_off = lane * fx.Int32(VEC_QK)
            v_off = lane * fx.Int32(VEC_V)
            q_row_base = (q_row * num_heads + head) * fx.Int32(QK_HEAD_DIM)
            q_f = [
                fx.BFloat16(q_buf[q_row_base + qk_off + i]).to(fx.Float32)
                for i in range_constexpr(VEC_QK)
            ]

            log2e = fx.Float32(_LOG2E)
            neg_inf = fx.Float32(_NEG_INF)
            c_zero = fx.Float32(0.0)
            bn = fx.Int32(BLOCK_N)

            init_state = [neg_inf, c_zero] + [c_zero for _ in range_constexpr(VEC_V)]
            final = init_state

            for j, state in range(0, kv_len, 1, init=init_state):
                m_cur = fx.Float32(state[0])
                l_cur = fx.Float32(state[1])
                a0 = fx.Float32(state[2])
                a1 = fx.Float32(state[3])
                a2 = fx.Float32(state[4])
                a3 = fx.Float32(state[5])
                a4 = fx.Float32(state[6])
                a5 = fx.Float32(state[7])
                a6 = fx.Float32(state[8])
                a7 = fx.Float32(state[9])

                j_i32 = fx.Int32(j)
                jl = j_i32 % bn
                n0 = j_i32 - jl

                def _stage_kv():
                    for e in range_constexpr(ELEMS_PER_THR):
                        flat = tid * fx.Int32(ELEMS_PER_THR) + fx.Int32(e)
                        row = flat // fx.Int32(QK_HEAD_DIM)
                        col = flat % fx.Int32(QK_HEAD_DIM)
                        jj = n0 + row
                        in_kv = jj < kv_len
                        page_idx = fx.Int32(kv_ix_buf[kv_start + jj // page_size])
                        phys = page_idx * page_size + (jj % page_size)
                        g_idx = phys * fx.Int32(QK_HEAD_DIM) + col
                        g_safe = in_kv.select(g_idx, fx.Int32(0))
                        val = in_kv.select(
                            fx.BFloat16(kv_buf[g_safe]), fx.BFloat16(0.0)
                        )
                        lds_kv[flat] = val
                    fx.gpu.barrier()

                if jl == fx.Int32(0):
                    _stage_kv()

                if const_expr(CAUSAL):
                    limit = kv_len - q_len + q_pos + fx.Int32(1)
                    valid = j_i32 < limit
                else:
                    valid = fx.Int32(1) == fx.Int32(1)

                lds_row = jl * fx.Int32(QK_HEAD_DIM)
                partial = c_zero
                for i in range_constexpr(VEC_QK):
                    k_f = fx.BFloat16(lds_kv[lds_row + qk_off + i]).to(fx.Float32)
                    partial = partial + q_f[i] * k_f

                score = partial
                score = score + score.shuffle_xor(fx.Int32(32), fx.Int32(64))
                score = score + score.shuffle_xor(fx.Int32(16), fx.Int32(64))
                score = score + score.shuffle_xor(fx.Int32(8), fx.Int32(64))
                score = score + score.shuffle_xor(fx.Int32(4), fx.Int32(64))
                score = score + score.shuffle_xor(fx.Int32(2), fx.Int32(64))
                score = score + score.shuffle_xor(fx.Int32(1), fx.Int32(64))
                score = score * sm_scale
                score = valid.select(score, neg_inf)

                m_new = m_cur.maximumf(score)
                was_empty = m_cur <= fx.Float32(_NEG_INF * 0.5)
                alpha = was_empty.select(c_zero, fmath.exp2((m_cur - m_new) * log2e))
                p = valid.select(fmath.exp2((score - m_new) * log2e), c_zero)
                l_new = l_cur * alpha + p

                def _acc(i, a_cur):
                    v_f = fx.BFloat16(lds_kv[lds_row + v_off + fx.Int32(i)]).to(
                        fx.Float32
                    )
                    return a_cur * alpha + p * valid.select(v_f, c_zero)

                na0 = _acc(0, a0)
                na1 = _acc(1, a1)
                na2 = _acc(2, a2)
                na3 = _acc(3, a3)
                na4 = _acc(4, a4)
                na5 = _acc(5, a5)
                na6 = _acc(6, a6)
                na7 = _acc(7, a7)

                last_in_tile = (jl == fx.Int32(BLOCK_N - 1)) | (
                    j_i32 + fx.Int32(1) == kv_len
                )
                if last_in_tile:
                    fx.gpu.barrier()

                final = yield [m_new, l_new, na0, na1, na2, na3, na4, na5, na6, na7]

            l_f = fx.Float32(final[1])
            has_kv = l_f > c_zero
            inv_l = has_kv.select(fx.Float32(1.0) / l_f, c_zero)
            o_row_base = (q_row * num_heads + head) * fx.Int32(V_HEAD_DIM)
            o_buf[o_row_base + v_off + 0] = (fx.Float32(final[2]) * inv_l).to(
                fx.BFloat16
            )
            o_buf[o_row_base + v_off + 1] = (fx.Float32(final[3]) * inv_l).to(
                fx.BFloat16
            )
            o_buf[o_row_base + v_off + 2] = (fx.Float32(final[4]) * inv_l).to(
                fx.BFloat16
            )
            o_buf[o_row_base + v_off + 3] = (fx.Float32(final[5]) * inv_l).to(
                fx.BFloat16
            )
            o_buf[o_row_base + v_off + 4] = (fx.Float32(final[6]) * inv_l).to(
                fx.BFloat16
            )
            o_buf[o_row_base + v_off + 5] = (fx.Float32(final[7]) * inv_l).to(
                fx.BFloat16
            )
            o_buf[o_row_base + v_off + 6] = (fx.Float32(final[8]) * inv_l).to(
                fx.BFloat16
            )
            o_buf[o_row_base + v_off + 7] = (fx.Float32(final[9]) * inv_l).to(
                fx.BFloat16
            )

        if active:
            _body()

    default_stream = fx.Stream(None)

    @flyc.jit
    def launch_mla_prefill_dense(
        q: fx.Tensor,
        kv: fx.Tensor,
        o: fx.Tensor,
        tile_q_start: fx.Tensor,
        tile_batch: fx.Tensor,
        qo_indptr: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_indices: fx.Tensor,
        page_size: fx.Int32,
        num_heads: fx.Int32,
        num_tiles: fx.Int32,
        sm_scale: fx.Float32,
        stream: fx.Stream = default_stream,
    ):
        kn_mla_prefill_dense(
            q,
            kv,
            o,
            tile_q_start,
            tile_batch,
            qo_indptr,
            kv_indptr,
            kv_indices,
            page_size,
            num_heads,
            num_tiles,
            sm_scale,
            value_attrs=kernel_value_attrs,
        ).launch(
            grid=(num_tiles, num_heads, 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )

    launch_mla_prefill_dense.compile_hints = dict(kernel_compile_hints)
    return launch_mla_prefill_dense


def _pad_sequences_to_block_m(q, o, qo_indptr, block_m: int = BLOCK_M):
    """Pad each sequence's Q/O length up to a multiple of block_m (full tiles)."""
    device = q.device
    batch = int(qo_indptr.numel()) - 1
    nhead, qk = q.shape[1], q.shape[2]
    vdim = o.shape[2]

    q_parts = []
    o_parts = []
    new_indptr = [0]
    real_lens = []
    for b in range(batch):
        s = int(qo_indptr[b].item())
        e = int(qo_indptr[b + 1].item())
        slen = e - s
        real_lens.append(slen)
        pad = (block_m - (slen % block_m)) % block_m
        q_parts.append(q[s:e])
        o_parts.append(o[s:e])
        if pad:
            q_parts.append(
                torch.zeros(pad, nhead, qk, dtype=q.dtype, device=device)
            )
            o_parts.append(
                torch.zeros(pad, nhead, vdim, dtype=o.dtype, device=device)
            )
        new_indptr.append(new_indptr[-1] + slen + pad)

    q_pad = torch.cat(q_parts, dim=0)
    o_pad = torch.cat(o_parts, dim=0)
    qo_pad = torch.tensor(new_indptr, dtype=torch.int32, device=device)
    return q_pad, o_pad, qo_pad, real_lens


def _build_q_tiles(qo_indptr: torch.Tensor, block_m: int = BLOCK_M):
    device = qo_indptr.device
    batch = int(qo_indptr.numel()) - 1
    starts: list[int] = []
    batches: list[int] = []
    for b in range(batch):
        s = int(qo_indptr[b].item())
        e = int(qo_indptr[b + 1].item())
        assert (e - s) % block_m == 0, "host must pad sequences to BLOCK_M"
        for q0 in range(s, e, block_m):
            starts.append(q0)
            batches.append(b)
    if not starts:
        z = torch.zeros(1, dtype=torch.int32, device=device)
        return z, z, 0
    return (
        torch.tensor(starts, dtype=torch.int32, device=device),
        torch.tensor(batches, dtype=torch.int32, device=device),
        len(starts),
    )


def flydsl_mla_prefill_dense_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float | None = None,
    *,
    is_causal: bool = False,
    block_n: int = DEFAULT_BLOCK_N,
    waves_per_eu: int = DEFAULT_WAVES_PER_EU,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Dense absorb MLA prefill (bf16). Writes and returns ``o``."""
    if q.dtype != torch.bfloat16 or kv_buffer.dtype != torch.bfloat16:
        raise TypeError("dense MLA prefill requires bf16 Q/KV")
    if o.dtype != torch.bfloat16:
        raise TypeError("output must be bf16")
    if q.ndim != 3 or o.ndim != 3:
        raise ValueError("q/o must be [total_q, H, dim]")

    total_q, num_heads, qk_dim = q.shape
    if qk_dim != QK_HEAD_DIM:
        raise ValueError(f"qk dim must be {QK_HEAD_DIM}, got {qk_dim}")
    if o.shape != (total_q, num_heads, V_HEAD_DIM):
        raise ValueError(
            f"o must be [{total_q}, {num_heads}, {V_HEAD_DIM}], got {tuple(o.shape)}"
        )

    if kv_buffer.ndim == 4:
        num_page, page_size, n_kv, kv_dim = kv_buffer.shape
        if n_kv != 1 or kv_dim != QK_HEAD_DIM:
            raise ValueError(
                f"kv expected [P, page, 1, {QK_HEAD_DIM}], got {tuple(kv_buffer.shape)}"
            )
        kv_flat = kv_buffer.reshape(num_page * page_size, QK_HEAD_DIM)
    elif kv_buffer.ndim == 3:
        if kv_buffer.shape[-1] != QK_HEAD_DIM or kv_buffer.shape[1] != 1:
            raise ValueError(f"bad kv 3D shape {tuple(kv_buffer.shape)}")
        page_size = 1
        kv_flat = kv_buffer.reshape(-1, QK_HEAD_DIM)
    elif kv_buffer.ndim == 2:
        if kv_buffer.shape[-1] != QK_HEAD_DIM:
            raise ValueError(f"bad kv 2D shape {tuple(kv_buffer.shape)}")
        page_size = 1
        kv_flat = kv_buffer
    else:
        raise ValueError(f"unsupported kv ndim {kv_buffer.ndim}")

    if sm_scale is None:
        sm_scale = 1.0 / (QK_HEAD_DIM**0.5)

    qo_indptr = qo_indptr.contiguous()
    kv_indptr = kv_indptr.contiguous()
    kv_indices = kv_indices.contiguous()
    q = q.contiguous()
    o = o.contiguous()

    q_pad, o_pad, qo_pad, real_lens = _pad_sequences_to_block_m(q, o, qo_indptr, BLOCK_M)
    tile_q_start, tile_batch, num_tiles = _build_q_tiles(qo_pad, BLOCK_M)
    if num_tiles == 0:
        return o

    launcher = compile_mla_prefill_dense(
        block_n=int(block_n),
        is_causal=bool(is_causal),
        waves_per_eu=int(waves_per_eu),
    )
    if stream is None:
        stream = torch.cuda.current_stream()

    launcher(
        q_pad.view(-1),
        kv_flat.contiguous().view(-1),
        o_pad.view(-1),
        tile_q_start,
        tile_batch,
        qo_pad,
        kv_indptr,
        kv_indices,
        int(page_size),
        int(num_heads),
        int(num_tiles),
        float(sm_scale),
        fx.Stream(stream),
    )

    # Scatter padded outputs back into the caller's o.
    batch = int(qo_indptr.numel()) - 1
    dst = 0
    src = 0
    for b in range(batch):
        slen = real_lens[b]
        o[dst : dst + slen].copy_(o_pad[src : src + slen])
        dst += slen
        pad = (BLOCK_M - (slen % BLOCK_M)) % BLOCK_M
        src += slen + pad
    return o
