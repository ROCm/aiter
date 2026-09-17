# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Family A FlyDSL QSA K2 (SILOTIGER-1047 3a): sparse GQA decode.

One workgroup per ``(row, kv_head)`` attends the expanded token list in the
paged K/V cache. Group size 12, ``D=256``. Online softmax in registers;
sigmoid gate and expand+tail stay unfused. Prefill occupancy is 3c.
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


def _idiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def _neg_inf():
    return Float32(float("-inf"))


def build_qsa_k2_family_a_module(page_size: int):
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if _D != _BLOCK_THREADS:
        raise ValueError("decode K2 maps one thread per D element")
    if _HQ != _HK * _GROUP:
        raise ValueError("family A GQA head counts do not form groups")

    @fx.struct
    class SharedStorage:
        q: fx.Array[BFloat16, _GROUP * _D, 16]
        k: fx.Array[BFloat16, _D, 16]
        v: fx.Array[BFloat16, _D, 16]
        red: fx.Array[Float32, 4, 16]
        m: fx.Array[Float32, _GROUP, 16]
        lse: fx.Array[Float32, _GROUP, 16]

    @flyc.kernel(
        name="qsa_k2_family_a_"
        + kernel_signature(
            ps=page_size,
            hq=_HQ,
            hk=_HK,
            d=_D,
            blk=_BLOCK_THREADS,
        ),
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def qsa_k2_family_a_kernel(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        v_cache: fx.Tensor,
        indices: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        out: fx.Tensor,
        n_sel: Int32,
        n_req: Int32,
        n_pages: Int32,
        softmax_scale: Float32,
    ):
        row = Int32(gpu.block_id("x"))
        kv_h = Int32(gpu.block_id("y"))
        tid = Int32(gpu.thread_id("x"))
        zero = Int32(0)
        one = Int32(1)
        page = Int32(page_size)
        elem_layout = fx.make_layout(1, 1)
        g_copy = buf_copy_atom(2, BFloat16)
        s_copy = fx.make_copy_atom(fx.UniversalCopy16b(), BFloat16)
        q_buf = fx.rocdl.make_buffer_tensor(q)
        k_buf = fx.rocdl.make_buffer_tensor(k_cache)
        v_buf = fx.rocdl.make_buffer_tensor(v_cache)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        smem_q = storage.q.view(fx.make_layout((_GROUP, _D), (_D, 1)))
        smem_k = storage.k.view(fx.make_layout(_D, 1))
        smem_v = storage.v.view(fx.make_layout(_D, 1))
        red = storage.red.view(fx.make_layout(4, 1))
        smem_m = storage.m.view(fx.make_layout(_GROUP, 1))
        smem_l = storage.lse.view(fx.make_layout(_GROUP, 1))

        req = token_to_req[row]
        valid_req = (req >= zero) & (req < n_req)
        safe_req = valid_req.select(req, zero)

        for h in range_constexpr(_GROUP):
            head = kv_h * Int32(_GROUP) + Int32(h)
            q_chunks = fx.logical_divide(
                fx.slice(q_buf, (row, head, None)), elem_layout
            )
            q_dst = fx.logical_divide(fx.slice(smem_q, (h, None)), elem_layout)
            q_src = fx.slice(q_chunks, (None, tid))
            q_sm = fx.slice(q_dst, (None, tid))
            q_frag = fx.make_fragment_like(q_src)
            fx.copy(g_copy, q_src, q_frag)
            fx.copy(s_copy, q_frag, q_sm)
            smem_m[h] = _neg_inf()
            smem_l[h] = Float32(0.0)
        gpu.barrier()

        init_acc = [Float32(0.0) for _h in range(_GROUP)]
        _start = fx.Int64(0)
        _stop = fx.Int64(n_sel)
        _step = fx.Int64(1)
        for col64, state in range(_start, _stop, _step, init=init_acc):
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
            k_dst = fx.logical_divide(smem_k, elem_layout)
            v_dst = fx.logical_divide(smem_v, elem_layout)
            k_src = fx.slice(k_chunks, (None, tid))
            v_src = fx.slice(v_chunks, (None, tid))
            k_sm = fx.slice(k_dst, (None, tid))
            v_sm = fx.slice(v_dst, (None, tid))
            k_frag = fx.make_fragment_like(k_src)
            v_frag = fx.make_fragment_like(v_src)
            fx.copy(g_copy, k_src, k_frag)
            fx.copy(s_copy, k_frag, k_sm)
            fx.copy(g_copy, v_src, v_frag)
            fx.copy(s_copy, v_frag, v_sm)
            gpu.barrier()
            k_f = smem_k[tid].to(Float32)
            v_f = smem_v[tid].to(Float32)
            new_acc = []
            for h in range_constexpr(_GROUP):
                acc = state[h]
                q_f = smem_q[h, tid].to(Float32)
                val = q_f * k_f
                for sh in (32, 16, 8, 4, 2, 1):
                    val = val + val.shuffle_xor(Int32(sh), Int32(64))
                red[_idiv(tid, Int32(64))] = val
                gpu.barrier()
                score = (
                    red[zero] + red[one] + red[Int32(2)] + red[Int32(3)]
                ) * softmax_scale
                gpu.barrier()
                m_prev = smem_m[h]
                l_prev = smem_l[h]
                m_new = m_prev.maximumf(score)
                alpha = fxmath.exp(m_prev - m_new)
                p = fxmath.exp(score - m_new)
                l_new = l_prev * alpha + p
                acc_new = acc * alpha + p * v_f
                acc_out = live.select(acc_new, acc)
                smem_m[h] = live.select(m_new, m_prev)
                smem_l[h] = live.select(l_new, l_prev)
                gpu.barrier()
                new_acc.append(acc_out)
            results = yield new_acc

        for h in range_constexpr(_GROUP):
            head = kv_h * Int32(_GROUP) + Int32(h)
            l_fin = smem_l[h]
            acc_h = results[h]
            out_f = (l_fin > Float32(0.0)).select(acc_h / l_fin, Float32(0.0))
            out[row, head, tid] = out_f.to(BFloat16)

    @flyc.jit
    def launch_qsa_k2_family_a(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        v_cache: fx.Tensor,
        indices: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        out: fx.Tensor,
        n_sel: Int32,
        n_req: Int32,
        n_pages: Int32,
        softmax_scale: Float32,
        rows: Int32,
        kv_heads: Int32,
        stream: fx.Stream,
    ):
        qsa_k2_family_a_kernel(
            q,
            k_cache,
            v_cache,
            indices,
            page_table,
            token_to_req,
            out,
            n_sel,
            n_req,
            n_pages,
            softmax_scale,
        ).launch(
            grid=(rows, kv_heads, 1),
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

    Attends ``indices [M, W]`` (``-1`` padded). Does not apply RoPE or the
    sigmoid gate. Expand+tail is still a separate launch.
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
    _run_compiled(
        _plan(page_size),
        q,
        k_cache,
        v_cache,
        indices,
        page_table,
        token_to_req,
        out,
        int(indices.shape[1]),
        int(page_table.shape[0]),
        int(page_table.shape[1]),
        float(softmax_scale),
        int(m),
        int(_HK),
        torch.cuda.current_stream(q.device),
    )
    return out
