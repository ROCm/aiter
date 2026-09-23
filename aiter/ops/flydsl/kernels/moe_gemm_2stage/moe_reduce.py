# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Device-cached routing reductions with no-dispatch AOT materialization."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, _run_compiled

from .. import moe_gemm_2stage_utils as fxh
from .common import device_context, get_device_cache_key
from .common import torch_tensor_to_pointer as _ptr


@functools.cache
def _sorted_sum_cached(device_cache_key, TOPK, N):
    del device_cache_key
    assert N % 256 == 0
    num_threads = 64 if N % 512 == 0 else 32

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def sorted_sum_kernel(loc_ids: fx.Pointer, A: fx.Pointer, B: fx.Pointer):
        batch = fx.block_idx.x
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
            (A + fx.Int64(token_locs[topk]) * N) for topk in fx.range_constexpr(TOPK)
        ]

        def load_atom(topk_id, off):
            atom = fxh.atom_tensor(token_ptrs[topk_id], fx.Int32(off), copy_bits)
            frag = fx.make_fragment_like(atom)
            fx.copy(copy_atom, atom, frag)
            return frag

        for dst, col in fxh.all_copy_atoms(
            B, col_tensor, atom_bits=copy_bits, num_threads=num_threads
        ):
            column = col[0].to_py_value()
            frag = [load_atom(topk, column) for topk in fx.range_constexpr(TOPK)]

            vec_sum = frag[0].load().to(fx.Float32)
            for m in fx.range_constexpr(1, TOPK):
                vec = frag[m].load().to(fx.Float32)
                vec_sum += vec

            vec_sum = vec_sum.to(dst.dtype)

            frag = fx.make_fragment_like(dst)
            frag.store(vec_sum)
            fx.copy(copy_atom_b, frag, dst)

    @flyc.jit
    def launch(
        loc_ids: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        assert A.dtype == B.dtype
        sorted_sum_kernel(loc_ids, A, B).launch(
            grid=(batch_size, 1, 1), block=(num_threads, 1, 1), stream=stream
        )

    def callable(
        loc_ids: torch.Tensor, A: torch.Tensor, B: torch.Tensor, batch_size: int
    ):
        with torch.cuda.device(B.device):
            _run_compiled(
                launch,
                _ptr(loc_ids),
                _ptr(A),
                _ptr(B),
                batch_size,
                torch.cuda.current_stream(B.device),
            )

    def precompile():
        _preload_compiled(
            launch,
            flyc.from_c_void_p(fx.Int32, 0),
            flyc.from_c_void_p(fx.BFloat16, 0),
            flyc.from_c_void_p(fx.BFloat16, 0),
            1,
            fx.Stream(None),
        )

    callable.precompile = precompile
    return callable


def sorted_sum(TOPK, N, *, device=None):
    return _sorted_sum_cached(get_device_cache_key(device), TOPK, N)


sorted_sum.cache_clear = _sorted_sum_cached.cache_clear
sorted_sum.cache_info = _sorted_sum_cached.cache_info


def compile_moe_reduction(*, topk, model_dim, device=None):
    """Return the baseline contiguous-row route reduction callable."""
    return sorted_sum(topk, model_dim, device=device)


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
        # Ignore the uninitialized tail beyond num_valid.
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
        stream: fx.Stream,
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
        with torch.cuda.device(sorted_ids.device):
            _run_compiled(
                launch,
                _ptr(sorted_ids),
                _ptr(invert),
                _ptr(num_valid),
                fx.Uint32(num_ids),
                fx.Uint32(batch_size),
                torch.cuda.current_stream(sorted_ids.device),
            )

    def precompile():
        _preload_compiled(
            launch,
            flyc.from_c_void_p(fx.Int32, 0),
            flyc.from_c_void_p(fx.Int32, 0),
            flyc.from_c_void_p(fx.Int32, 0),
            fx.Uint32(1),
            fx.Uint32(1),
            fx.Stream(None),
        )

    callable.precompile = precompile
    return callable


def invert_sorted_ids(TOPK, *, device=None):
    return _invert_sorted_ids_cached(get_device_cache_key(device), TOPK)


invert_sorted_ids.cache_clear = _invert_sorted_ids_cached.cache_clear
invert_sorted_ids.cache_info = _invert_sorted_ids_cached.cache_info


def precompile_moe_reduction_kernels(topk, model_dim, *, device=None):
    """Preload both BF16 prefill reduction launchers without executing them."""
    with device_context(device):
        invert_sorted_ids(topk, device=device).precompile()
        sorted_sum(topk, model_dim, device=device).precompile()
