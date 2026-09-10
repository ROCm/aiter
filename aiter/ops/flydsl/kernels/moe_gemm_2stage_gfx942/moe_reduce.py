# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from . import layout_helpers as fxh
from .common import get_device_cache_key
from .common import torch_tensor_to_pointer as _ptr


@functools.cache
def _sorted_sum_cached(
    device_cache_key,
    TOPK,
    N,
    row_padding_bytes=None,
):
    del device_cache_key
    if row_padding_bytes is not None:
        assert row_padding_bytes in (0, 32, 64, 128)
    num_threads = 64
    source_row_stride = N + (
        row_padding_bytes // 2 if row_padding_bytes is not None else 0
    )

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def sorted_sum_kernel(loc_ids: fx.Pointer, A: fx.Pointer, B: fx.Pointer):
        batch = fx.block_idx.x
        # preload all TOPK locations
        loc_ids += batch * TOPK
        token_locs = [loc_ids[topk] for topk in fx.range_constexpr(TOPK)]

        copy_bits = 128
        copy_atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), A.dtype)
        copy_atom_b = fx.make_copy_atom(fx.rocdl.BufferCopy(copy_bits), B.dtype)

        col_tensor = fx.make_view(0, fx.make_layout(N, 1))
        B = fx.make_view(B + fx.Int64(batch) * (N), fx.make_layout(N, 1))
        B = fx.rocdl.make_buffer_tensor(
            B,
            max_size=False,
            num_records_bytes=fx.Int64(N) * (B.dtype.width // 8),
        )

        token_ptrs = [
            (A + fx.Int64(token_locs[topk]) * source_row_stride)
            for topk in fx.range_constexpr(TOPK)
        ]

        def load_atom(topk_id, off):
            atom = fxh.atom_tensor(token_ptrs[topk_id], fx.Int32(off), copy_bits)
            frag = fx.make_fragment_like(atom)
            fx.copy(copy_atom, atom, frag)
            return frag

        def load_atoms(off):
            return [load_atom(topk, off) for topk in fx.range_constexpr(TOPK)]

        def reduce_store(dst, frag):
            vec_sum = frag[0].load().to(fx.Float32)
            for m in fx.range_constexpr(1, TOPK):
                vec_sum += frag[m].load().to(fx.Float32)
            vec_sum = vec_sum.to(dst.dtype)
            out_frag = fx.make_fragment_like(dst)
            out_frag.store(vec_sum)
            fx.copy(copy_atom_b, out_frag, dst)

        for dst, col in fxh.all_copy_atoms(
            B,
            col_tensor,
            atom_bits=copy_bits,
            num_threads=num_threads,
        ):
            reduce_store(dst, load_atoms(col[0].to_py_value()))

    @flyc.jit
    def launch(
        loc_ids: fx.Pointer, A: fx.Pointer, B: fx.Pointer, batch_size: fx.Int32, stream
    ):
        assert A.dtype == B.dtype
        sorted_sum_kernel(loc_ids, A, B).launch(
            grid=(batch_size, 1, 1), block=(num_threads, 1, 1), stream=stream
        )

    def callable(
        loc_ids: torch.Tensor, A: torch.Tensor, B: torch.Tensor, batch_size: int
    ):
        stream = torch.cuda.current_stream()
        _run_compiled(
            launch,
            _ptr(loc_ids),
            _ptr(A),
            _ptr(B),
            batch_size,
            stream,
        )

    return callable


def sorted_sum(TOPK, N, row_padding_bytes=None):
    return _sorted_sum_cached(get_device_cache_key(), TOPK, N, row_padding_bytes)


sorted_sum.cache_clear = _sorted_sum_cached.cache_clear
sorted_sum.cache_info = _sorted_sum_cached.cache_info


def compile_moe_reduction(*, topk, model_dim, row_padding_bytes=None):
    """Build and cache the route reduction callable for one static configuration."""
    return sorted_sum(topk, model_dim, row_padding_bytes)


@functools.cache
def _invert_sorted_ids_cached(device_cache_key, TOPK):
    del device_cache_key
    num_threads = 64

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def invert_sorted_ids_kernel(
        sorted_ids: fx.Pointer,
        invert: fx.Pointer,
        p_num_valid: fx.Pointer,
        num_ids: fx.Uint32,
        batch_size: fx.Uint32,
    ):
        batch = fx.block_idx.x
        tid = fx.thread_idx.x
        slot = batch * num_threads + tid
        # Scan only the down-written region [0, num_valid): the tail of sorted_ids
        # (>= num_valid) is uninitialized, and its garbage would racily map real
        # tokens onto unwritten gemm2_out rows. Read the bound on-device (no host sync).
        num_valid = p_num_valid[0].to(fx.Uint32)
        if slot < num_valid:
            sid = sorted_ids[slot].to(fx.Uint32)
            tok_id = sid & 0xFFFFFF
            top_id = sid >> 24
            idx = tok_id * TOPK + top_id
            if top_id < TOPK and tok_id < batch_size:
                invert[idx] = fx.Uint32(slot)

    @flyc.jit
    def launch(
        sorted_ids: fx.Pointer,
        invert: fx.Pointer,
        p_num_valid: fx.Pointer,
        num_ids: fx.Uint32,
        batch_size: fx.Uint32,
        stream,
    ):
        grid_size = fxh.div_up(num_ids, num_threads)
        invert_sorted_ids_kernel(
            sorted_ids, invert, p_num_valid, num_ids, batch_size
        ).launch(grid=(grid_size, 1, 1), block=(num_threads, 1, 1), stream=stream)

    def callable(
        sorted_ids: torch.Tensor,
        invert: torch.Tensor,
        num_valid: torch.Tensor,
        num_ids: int,
        batch_size: int,
    ):
        stream = torch.cuda.current_stream()
        _run_compiled(
            launch,
            _ptr(sorted_ids),
            _ptr(invert),
            _ptr(num_valid),
            fx.Uint32(num_ids),
            fx.Uint32(batch_size),
            stream,
        )

    return callable


def invert_sorted_ids(TOPK):
    return _invert_sorted_ids_cached(get_device_cache_key(), TOPK)


invert_sorted_ids.cache_clear = _invert_sorted_ids_cached.cache_clear
invert_sorted_ids.cache_info = _invert_sorted_ids_cached.cache_info
