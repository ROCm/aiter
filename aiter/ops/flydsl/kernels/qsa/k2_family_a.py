# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Family A FlyDSL QSA K2 (SILOTIGER-1047 3c): sparse GQA with split-K.

One workgroup per ``(row, kv_head, split)`` attends a slice of the expanded
token list in the paged K/V cache, then a merge kernel combines the partial
softmax states. Group size 12, ``D=256``. Decode and prefill share this
instantiation. Sigmoid gate and expand+tail stay unfused.
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
    """Match live AMD decode occupancy: many splits when ``M * Hk`` is small."""
    if n_sel < 1:
        return 1
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
        target = 1
    return max(1, min(target, n_sel))


def build_qsa_k2_family_a_module(page_size: int):
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if _D != _BLOCK_THREADS:
        raise ValueError("decode K2 maps one thread per D element")
    if _HQ != _HK * _GROUP:
        raise ValueError("family A GQA head counts do not form groups")

    @fx.struct
    class SharedStorage:
        red: fx.Array[Float32, _GROUP * 4, 16]

    @flyc.kernel(
        name="qsa_k2_family_a_split_"
        + kernel_signature(
            ps=page_size,
            hq=_HQ,
            hk=_HK,
            d=_D,
            blk=_BLOCK_THREADS,
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
        elem_layout = fx.make_layout(1, 1)
        g_copy = buf_copy_atom(2, BFloat16)
        q_buf = fx.rocdl.make_buffer_tensor(q)
        k_buf = fx.rocdl.make_buffer_tensor(k_cache)
        v_buf = fx.rocdl.make_buffer_tensor(v_cache)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        red = storage.red.view(fx.make_layout((_GROUP, 4), (4, 1)))

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

        init_state = (
            [_neg_inf() for _h in range(_GROUP)]
            + [Float32(0.0) for _h in range(_GROUP)]
            + [Float32(0.0) for _h in range(_GROUP)]
        )
        _start = fx.Int64(col_start)
        _stop = fx.Int64(col_end)
        _step = fx.Int64(1)
        for col64, state in range(_start, _stop, _step, init=init_state):
            col = Int32(col64)
            tok = indices[row, col]
            live = valid_req & (tok >= zero)
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
            k_f = fx.Vector(fx.memref_load_vec(k_frag))[0].to(Float32)
            v_f = fx.Vector(fx.memref_load_vec(v_frag))[0].to(Float32)
            for h in range_constexpr(_GROUP):
                val = q_regs[h] * k_f
                for sh in (32, 16, 8, 4, 2, 1):
                    val = val + val.shuffle_xor(Int32(sh), Int32(64))
                red[h, wave] = val
            gpu.barrier()
            new_m = []
            new_l = []
            new_a = []
            for h in range_constexpr(_GROUP):
                score = (
                    red[h, zero] + red[h, one] + red[h, Int32(2)] + red[h, Int32(3)]
                ) * softmax_scale
                m_prev = state[h]
                l_prev = state[_GROUP + h]
                acc = state[2 * _GROUP + h]
                m_new = m_prev.maximumf(score)
                alpha = fxmath.exp(m_prev - m_new)
                p = fxmath.exp(score - m_new)
                l_new = l_prev * alpha + p
                acc_new = acc * alpha + p * v_f
                new_m.append(live.select(m_new, m_prev))
                new_l.append(live.select(l_new, l_prev))
                new_a.append(live.select(acc_new, acc))
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

    Attends ``indices [M, W]`` (``-1`` padded). Split-K plus LSE merge when
    decode occupancy is low. Does not apply RoPE or the sigmoid gate.
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
