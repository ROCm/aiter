# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Family A FlyDSL QSA K2 (SILOTIGER-1047 3d): tiled sparse GQA.

One workgroup per ``(row, kv_head, split)`` gathers ``BLOCK_N`` paged K/V
columns, scores them with ``MFMA 16x16x16`` (group padded to 16), and
updates online softmax on that tile. Split-K plus LSE merge is unchanged.
Decode and prefill share this instantiation. Sigmoid and expand+tail stay
unfused.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import BFloat16, Float32, Int32, gpu, range_constexpr
from flydsl.expr import math as fxmath

from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from aiter.ops.flydsl.kernels.qsa.shapes import FAMILY_A_GQA
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, buf_copy_atom

_BLOCK_THREADS = 256
_BLOCK_N = 16
_HQ = FAMILY_A_GQA.n_heads
_HK = FAMILY_A_GQA.kv_heads
_GROUP = FAMILY_A_GQA.group_size
_D = FAMILY_A_GQA.head_dim
_DEFAULT_SCALE = _D**-0.5
_LSE_EMPTY = -1.0e20


def _idiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def _neg_inf():
    return Float32(float("-inf"))


def _choose_splits(rows: int, n_sel: int) -> int:
    """Keep decode occupancy; cap prefill serial tiles at ~4 per split."""
    if n_sel < 1:
        return 1
    tiles = (n_sel + _BLOCK_N - 1) // _BLOCK_N
    base = rows * _HK
    if base <= 8:
        target = 64
    elif base < 32:
        target = 32
    elif base <= 256:
        target = 8
    elif base <= 512:
        target = 4
    else:
        target = max(1, tiles // 4)
        target = min(32, target)
    return max(1, min(target, tiles, n_sel))


def build_qsa_k2_family_a_module(page_size: int):
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if _D != _BLOCK_THREADS:
        raise ValueError("decode K2 maps one thread per D element")
    if _HQ != _HK * _GROUP:
        raise ValueError("family A GQA head counts do not form groups")

    _HEAD_PAD = 16
    _K_STEPS = _D // 64  # 4 MFMA K-steps per wave's 64-wide D slice

    @fx.struct
    class SharedStorage:
        q: fx.Array[BFloat16, _HEAD_PAD * _D, 16]
        k: fx.Array[BFloat16, _BLOCK_N * _D, 16]
        s: fx.Array[Float32, _HEAD_PAD * _BLOCK_N, 16]
        c: fx.Array[Float32, 4 * 64 * 4, 16]

    @flyc.kernel(
        name="qsa_k2_family_a_split_"
        + kernel_signature(
            ps=page_size,
            hq=_HQ,
            hk=_HK,
            d=_D,
            blk=_BLOCK_THREADS,
            bn=_BLOCK_N,
            mm=16,
        ),
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def qsa_k2_family_a_split_kernel(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        v_cache: fx.Tensor,
        indices: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        partial_out: fx.Tensor,
        partial_lse: fx.Tensor,
        n_sel: Int32,
        n_req: Int32,
        n_pages: Int32,
        n_splits: Int32,
        softmax_scale: Float32,
    ):
        row = Int32(gpu.block_id("x"))
        kv_h = Int32(gpu.block_id("y"))
        split = Int32(gpu.block_id("z"))
        tid = Int32(gpu.thread_id("x"))
        zero = Int32(0)
        one = Int32(1)
        page = Int32(page_size)
        wave = _idiv(tid, Int32(64))
        lane = tid - wave * Int32(64)
        lane_m = lane % Int32(16)
        lane_kg = _idiv(lane, Int32(16))
        elem_layout = fx.make_layout(1, 1)
        g_copy = buf_copy_atom(2, BFloat16)
        q_buf = fx.rocdl.make_buffer_tensor(q)
        k_buf = fx.rocdl.make_buffer_tensor(k_cache)
        v_buf = fx.rocdl.make_buffer_tensor(v_cache)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        q_lds = storage.q.view(fx.make_layout((_HEAD_PAD, _D), (_D, 1)))
        k_lds = storage.k.view(fx.make_layout((_BLOCK_N, _D), (_D, 1)))
        s_lds = storage.s.view(fx.make_layout((_HEAD_PAD, _BLOCK_N), (_BLOCK_N, 1)))
        c_lds = storage.c.view(fx.make_layout((4, 64, 4), (64 * 4, 4, 1)))
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, BFloat16))
        fa = fx.make_rmem_tensor(fx.make_layout(4, 1), BFloat16)
        fb = fx.make_rmem_tensor(fx.make_layout(4, 1), BFloat16)
        fc = fx.make_rmem_tensor(fx.make_layout(4, 1), Float32)

        def mfma_acc(a_vec, b_vec, c_vec):
            fx.memref_store_vec(a_vec, fa)
            fx.memref_store_vec(b_vec, fb)
            fx.memref_store_vec(c_vec, fc)
            fx.mma_atom_call(mma, fc, fa, fb, fc)
            return fx.memref_load_vec(fc)

        req = token_to_req[row]
        valid_req = (req >= zero) & (req < n_req)
        safe_req = valid_req.select(req, zero)
        col_start = _idiv(split * n_sel, n_splits)
        col_end = _idiv((split + one) * n_sel, n_splits)

        q_regs = []
        for h in range_constexpr(_GROUP):
            head = kv_h * Int32(_GROUP) + Int32(h)
            q_chunks = fx.logical_divide(
                fx.slice(q_buf, (row, head, None)), elem_layout
            )
            q_src = fx.slice(q_chunks, (None, tid))
            q_frag = fx.make_fragment_like(q_src)
            fx.copy(g_copy, q_src, q_frag)
            q_regs.append(fx.Vector(fx.memref_load_vec(q_frag))[0].to(Float32))
        for h in range_constexpr(_HEAD_PAD):
            qv = q_regs[h] if h < _GROUP else Float32(0.0)
            q_lds[h, tid] = qv.to(BFloat16)
        gpu.barrier()

        init_state = (
            [_neg_inf() for _h in range(_GROUP)]
            + [Float32(0.0) for _h in range(_GROUP)]
            + [Float32(0.0) for _h in range(_GROUP)]
        )
        span = col_end - col_start
        n_tiles = _idiv(span + Int32(_BLOCK_N - 1), Int32(_BLOCK_N))
        _start = fx.Int64(0)
        _stop = fx.Int64(n_tiles)
        _step = fx.Int64(1)
        for t64, state in range(_start, _stop, _step, init=init_state):
            base = col_start + Int32(t64) * Int32(_BLOCK_N)
            k_regs = []
            v_regs = []
            live_n = []
            for n in range_constexpr(_BLOCK_N):
                col = base + Int32(n)
                in_col = col < col_end
                safe_col = in_col.select(col, col_start)
                tok = indices[row, safe_col]
                live = valid_req & in_col & (tok >= zero)
                safe_tok = (tok >= zero).select(tok, zero)
                logical_page = _idiv(safe_tok, page)
                off = safe_tok - logical_page * page
                in_table = logical_page < n_pages
                page_idx = in_table.select(logical_page, zero)
                phys = page_table[safe_req, page_idx]
                k_chunks = fx.logical_divide(
                    fx.slice(k_buf, (phys, off, kv_h, None)), elem_layout
                )
                v_chunks = fx.logical_divide(
                    fx.slice(v_buf, (phys, off, kv_h, None)), elem_layout
                )
                k_src = fx.slice(k_chunks, (None, tid))
                v_src = fx.slice(v_chunks, (None, tid))
                k_frag = fx.make_fragment_like(k_src)
                v_frag = fx.make_fragment_like(v_src)
                fx.copy(g_copy, k_src, k_frag)
                fx.copy(g_copy, v_src, v_frag)
                k_regs.append(fx.Vector(fx.memref_load_vec(k_frag))[0].to(Float32))
                v_regs.append(fx.Vector(fx.memref_load_vec(v_frag))[0].to(Float32))
                live_n.append(live)
            for n in range_constexpr(_BLOCK_N):
                k_lds[n, tid] = k_regs[n].to(BFloat16)
            gpu.barrier()
            acc4 = fx.Vector.filled(4, 0.0, Float32)
            for ks in range_constexpr(_K_STEPS):
                d0 = wave * Int32(64) + Int32(ks * 16) + lane_kg * Int32(4)
                a_vec = fx.Vector.from_elements(
                    [
                        q_lds[lane_m, d0],
                        q_lds[lane_m, d0 + one],
                        q_lds[lane_m, d0 + Int32(2)],
                        q_lds[lane_m, d0 + Int32(3)],
                    ],
                    BFloat16,
                )
                b_vec = fx.Vector.from_elements(
                    [
                        k_lds[lane_m, d0],
                        k_lds[lane_m, d0 + one],
                        k_lds[lane_m, d0 + Int32(2)],
                        k_lds[lane_m, d0 + Int32(3)],
                    ],
                    BFloat16,
                )
                acc4 = fx.Vector(mfma_acc(a_vec, b_vec, acc4))
            for i in range_constexpr(4):
                c_lds[wave, lane, i] = acc4[i]
            gpu.barrier()
            # CDNA 16x16x16 C fragment: C[i] is S[4*(lane//16)+i, lane%16].
            for i in range_constexpr(4):
                s_i = (
                    c_lds[zero, lane, i]
                    + c_lds[one, lane, i]
                    + c_lds[Int32(2), lane, i]
                    + c_lds[Int32(3), lane, i]
                )
                s_lds[lane_kg * Int32(4) + Int32(i), lane_m] = s_i
            gpu.barrier()
            any_live = live_n[0]
            for n in range_constexpr(1, _BLOCK_N):
                any_live = any_live | live_n[n]
            new_m = []
            new_l = []
            new_a = []
            for h in range_constexpr(_GROUP):
                scores = []
                for n in range_constexpr(_BLOCK_N):
                    score = s_lds[h, n] * softmax_scale
                    scores.append(live_n[n].select(score, _neg_inf()))
                tile_max = scores[0]
                for n in range_constexpr(1, _BLOCK_N):
                    tile_max = tile_max.maximumf(scores[n])
                m_prev = state[h]
                l_prev = state[_GROUP + h]
                acc = state[2 * _GROUP + h]
                m_new = m_prev.maximumf(tile_max)
                alpha = fxmath.exp(m_prev - m_new)
                p_sum = Float32(0.0)
                acc_add = Float32(0.0)
                for n in range_constexpr(_BLOCK_N):
                    p = live_n[n].select(fxmath.exp(scores[n] - m_new), Float32(0.0))
                    p_sum = p_sum + p
                    acc_add = acc_add + p * v_regs[n]
                l_new = l_prev * alpha + p_sum
                acc_new = acc * alpha + acc_add
                new_m.append(any_live.select(m_new, m_prev))
                new_l.append(any_live.select(l_new, l_prev))
                new_a.append(any_live.select(acc_new, acc))
            gpu.barrier()
            results = yield new_m + new_l + new_a

        for h in range_constexpr(_GROUP):
            head = kv_h * Int32(_GROUP) + Int32(h)
            m_fin = results[h]
            l_fin = results[_GROUP + h]
            acc_h = results[2 * _GROUP + h]
            has = l_fin > Float32(0.0)
            out_f = has.select(acc_h / l_fin, Float32(0.0))
            lse = has.select(m_fin + fxmath.log(l_fin), Float32(_LSE_EMPTY))
            partial_out[split, row, head, tid] = out_f
            partial_lse[split, row, head] = lse

    @flyc.kernel(
        name="qsa_k2_family_a_merge_"
        + kernel_signature(hq=_HQ, d=_D, blk=_BLOCK_THREADS),
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def qsa_k2_family_a_merge_kernel(
        partial_out: fx.Tensor,
        partial_lse: fx.Tensor,
        out: fx.Tensor,
        n_splits: Int32,
    ):
        row = Int32(gpu.block_id("x"))
        head = Int32(gpu.block_id("y"))
        tid = Int32(gpu.thread_id("x"))
        m_max = Float32(_LSE_EMPTY)
        zero_f = Float32(0.0)
        _start = fx.Int64(0)
        _stop = fx.Int64(n_splits)
        _step = fx.Int64(1)
        for s64, state in range(_start, _stop, _step, init=[m_max, zero_f]):
            s = Int32(s64)
            m_in = state[0]
            lse = partial_lse[s, row, head]
            results = yield [m_in.maximumf(lse), zero_f]
        m_max = results[0]
        acc = Float32(0.0)
        den = Float32(0.0)
        for s64, state in range(_start, _stop, _step, init=[acc, den]):
            s = Int32(s64)
            acc_in = state[0]
            den_in = state[1]
            lse = partial_lse[s, row, head]
            live = lse > Float32(_LSE_EMPTY)
            w = live.select(fxmath.exp(lse - m_max), Float32(0.0))
            part = partial_out[s, row, head, tid]
            results = yield [acc_in + w * part, den_in + w]
        acc_f = Float32(results[0])
        den_f = Float32(results[1])
        out_f = (den_f > Float32(0.0)).select(acc_f / den_f, Float32(0.0))
        out[row, head, tid] = out_f.to(BFloat16)

    @flyc.jit
    def launch_qsa_k2_family_a(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        v_cache: fx.Tensor,
        indices: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        partial_out: fx.Tensor,
        partial_lse: fx.Tensor,
        out: fx.Tensor,
        n_sel: Int32,
        n_req: Int32,
        n_pages: Int32,
        n_splits: Int32,
        softmax_scale: Float32,
        rows: Int32,
        kv_heads: Int32,
        n_heads: Int32,
        stream: fx.Stream,
    ):
        qsa_k2_family_a_split_kernel(
            q,
            k_cache,
            v_cache,
            indices,
            page_table,
            token_to_req,
            partial_out,
            partial_lse,
            n_sel,
            n_req,
            n_pages,
            n_splits,
            softmax_scale,
        ).launch(
            grid=(rows, kv_heads, n_splits),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        qsa_k2_family_a_merge_kernel(
            partial_out,
            partial_lse,
            out,
            n_splits,
        ).launch(
            grid=(rows, n_heads, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_qsa_k2_family_a


@lru_cache(maxsize=8)
def _plan(page_size: int):
    return build_qsa_k2_family_a_module(page_size)


def qsa_k2_family_a_serves(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    page_table: torch.Tensor,
) -> str | None:
    """Why this K2 kernel cannot serve these tensors, or None if it can."""
    gqa = FAMILY_A_GQA
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        return f"q and caches must be bfloat16, got {q.dtype} and {k_cache.dtype}"
    if v_cache.dtype != k_cache.dtype or v_cache.shape != k_cache.shape:
        return "v_cache must match k_cache dtype and shape"
    if q.dim() != 3 or q.shape[1] != gqa.n_heads or q.shape[2] != gqa.head_dim:
        return f"q must be [M, {gqa.n_heads}, {gqa.head_dim}], got {tuple(q.shape)}"
    if k_cache.dim() != 4:
        return f"k_cache must be [pages, page_size, H, D], got {tuple(k_cache.shape)}"
    if k_cache.shape[2] != gqa.kv_heads or k_cache.shape[3] != gqa.head_dim:
        return (
            f"k_cache KV/D must be ({gqa.kv_heads}, {gqa.head_dim}), "
            f"got {k_cache.shape[2:]}"
        )
    if indices.dim() != 2 or indices.shape[0] != q.shape[0]:
        return "indices must be [M, W]"
    if indices.dtype != torch.int32:
        return f"indices must be int32, got {indices.dtype}"
    if page_table.dim() != 2 or page_table.dtype != torch.int32:
        return (
            f"page_table must be int32 [n_req, n_pages], got {tuple(page_table.shape)}"
        )
    return None


def qsa_k2_family_a(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Write family A sparse GQA ``o [M, 24, 256]`` from paged K/V.

    Attends ``indices [M, W]`` (``-1`` padded). Tiled ``BLOCK_N`` MFMA QK with
    split-K plus LSE merge. Does not apply RoPE or the sigmoid gate.
    Expand+tail is still a separate launch.
    """
    reason = qsa_k2_family_a_serves(q, k_cache, v_cache, indices, page_table)
    if reason is not None:
        raise ValueError(f"[FlyDSL qsa_k2_family_a] {reason}")
    m = q.shape[0]
    if token_to_req.shape != (m,) or token_to_req.dtype != torch.int32:
        raise ValueError(f"token_to_req must be int32 [{m}]")
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype:
        raise ValueError(f"out must match q, got {tuple(out.shape)} {out.dtype}")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    tensors = (q, k_cache, v_cache, indices, page_table, token_to_req, out)
    if any(not t.is_cuda for t in tensors):
        raise ValueError("every tensor must be on the GPU")
    if any(t.device != q.device for t in tensors[1:]):
        raise ValueError("every tensor must be on the same GPU")
    if softmax_scale is None:
        softmax_scale = _DEFAULT_SCALE
    if not m or not indices.shape[1]:
        return out.zero_()
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    indices = indices.contiguous()
    page_table = page_table.contiguous()
    token_to_req = token_to_req.contiguous()
    page_size = k_cache.shape[1]
    n_sel = int(indices.shape[1])
    n_splits = _choose_splits(m, n_sel)
    partial_out = torch.empty(
        (n_splits, m, _HQ, _D), dtype=torch.float32, device=q.device
    )
    partial_lse = torch.empty((n_splits, m, _HQ), dtype=torch.float32, device=q.device)
    _run_compiled(
        _plan(page_size),
        q,
        k_cache,
        v_cache,
        indices,
        page_table,
        token_to_req,
        partial_out,
        partial_lse,
        out,
        n_sel,
        int(page_table.shape[0]),
        int(page_table.shape[1]),
        n_splits,
        float(softmax_scale),
        int(m),
        int(_HK),
        int(_HQ),
        torch.cuda.current_stream(q.device),
    )
    return out
