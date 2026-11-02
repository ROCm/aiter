# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routing reductions with cached JIT launchers."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from . import common as fxh
from .common import _ptr


@functools.cache
def sorted_sum(TOPK, N, output_padding_bytes=0):
    assert N % 256 == 0
    assert output_padding_bytes in (0, 32, 64, 128)
    input_row_stride = N + output_padding_bytes // 2
    num_threads = 64 if N % 512 == 0 else 32

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def sorted_sum_kernel(loc_ids: fx.Pointer, A: fx.Pointer, B: fx.Pointer):
        batch = fx.block_idx.x
        loc_ids += batch * TOPK
        token_locs = [loc_ids[topk] for topk in fx.range_constexpr(TOPK)]
        # Sorting gives -1 to routes merged as repeated expert IDs of a token.
        valid = [loc >= 0 for loc in token_locs]

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
            (A + fx.Int64(valid[topk].select(token_locs[topk], 0)) * input_row_stride)
            for topk in fx.range_constexpr(TOPK)
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

            vec_sum = valid[0].select(frag[0].load().to(fx.Float32), 0.0)
            for m in fx.range_constexpr(1, TOPK):
                vec_sum += valid[m].select(frag[m].load().to(fx.Float32), 0.0)

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
        stream,
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
