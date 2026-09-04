# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL split merge for the gfx1250 MLA decode."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T

from ..mla_reduce import (
    _exp,
    _load_partial_out,
    _out_numeric_t,
    _pointer_buffer_tensor,
    _store_final_out,
)

NUM_THREADS = 128
LOG2_PAGE_SIZE = 6
_DEFAULT_WAVES_PER_EU = 4
_DEFAULT_STREAM = fx.Stream(None)


@functools.lru_cache(maxsize=32)
def compile_mla_decode_reduce(
    *,
    H: int,
    Dv: int,
    out_dtype: str = "bf16",
    waves_per_eu: int = _DEFAULT_WAVES_PER_EU,
):
    assert (
        Dv % NUM_THREADS == 0
    ), f"Dv ({Dv}) must be divisible by NUM_THREADS ({NUM_THREADS})"
    VEC = Dv // NUM_THREADS

    kernel_value_attrs = (
        {"rocdl.waves_per_eu": int(waves_per_eu)} if waves_per_eu >= 1 else {}
    )
    kernel_compile_hints = (
        {"waves_per_eu": int(waves_per_eu)} if waves_per_eu >= 1 else {}
    )

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def mla_decode_reduce_kernel(
        split_data: fx.Pointer,  # fp32 [total_tokens * num_splits, H, Dv]
        split_lse: fx.Pointer,  # fp32 [total_tokens * num_splits, H]
        seq_lens: fx.Pointer,  # i32  [num_seqs]
        final_output: fx.Pointer,  # out  [total_tokens, H, Dv]
        total_tokens: fx.Int32,
        num_seqs: fx.Int32,
        num_splits: fx.Int32,
        num_tokens_per_seq: fx.Int32,
    ):
        out_numeric_t = _out_numeric_t(out_dtype)

        tid = fx.thread_idx.x
        head = fx.block_idx.x
        token = fx.block_idx.y

        num_rows = total_tokens * num_splits
        data_buf = _pointer_buffer_tensor(
            split_data, fx.Float32, (num_rows, H, Dv), (H * Dv, Dv, 1)
        )
        lse_buf = _pointer_buffer_tensor(split_lse, fx.Float32, (num_rows, H), (H, 1))
        out_buf = _pointer_buffer_tensor(
            final_output, out_numeric_t, (total_tokens, H, Dv), (H * Dv, Dv, 1)
        )
        seq_buf = _pointer_buffer_tensor(seq_lens, fx.Int32, (num_seqs,), (1,))

        token_i32 = fx.Int32(token)
        seq_id = token_i32 // num_tokens_per_seq
        token_in_seq = token_i32 - seq_id * num_tokens_per_seq
        seq_len = seq_buf[seq_id] - num_tokens_per_seq + token_in_seq + fx.Int32(1)
        num_pages = (seq_len + fx.Int32((1 << LOG2_PAGE_SIZE) - 1)) >> fx.Int32(
            LOG2_PAGE_SIZE
        )
        over = num_splits > num_pages
        valid_splits = over.select(num_pages, num_splits)

        row_base = token_i32 * num_splits

        partial0 = _load_partial_out(data_buf, row_base, head, tid, VEC)
        lse0 = lse_buf[row_base, head]
        init = [partial0[i].ir_value() for i in fx.range_constexpr(VEC)]
        init += [fx.Float32(lse0).ir_value(), fx.Float32(1.0).ir_value()]

        results = init
        for split, state in range(fx.Int32(1), valid_splits, fx.Int32(1), init=init):
            acc = [state[i] for i in fx.range_constexpr(VEC)]
            running_max = state[VEC]
            running_sum = state[VEC + 1]

            row = row_base + fx.Int32(split)
            partial = _load_partial_out(data_buf, row, head, tid, VEC)
            lse = lse_buf[row, head]

            new_max = fx.Float32(running_max).maximumf(lse)
            rescale = _exp(fx.Float32(running_max) - new_max)
            weight = _exp(lse - new_max)
            new_acc = [
                (fx.Float32(acc[i]) * rescale + partial[i] * weight).ir_value()
                for i in fx.range_constexpr(VEC)
            ]
            results = yield new_acc + [
                new_max.ir_value(),
                (fx.Float32(running_sum) * rescale + weight).ir_value(),
            ]

        acc = [results[i] for i in fx.range_constexpr(VEC)]
        running_sum = fx.Float32(results[VEC + 1])
        inv_sum = fx.rocdl.rcp(T.f32, running_sum.ir_value())
        out_elems = [fx.Float32(acc[i]) * inv_sum for i in fx.range_constexpr(VEC)]
        _store_final_out(out_buf, token, head, tid, out_elems, VEC, out_numeric_t)

    @flyc.jit
    def launch_mla_decode_reduce(
        split_data: fx.Pointer,
        split_lse: fx.Pointer,
        seq_lens: fx.Pointer,
        final_output: fx.Pointer,
        total_tokens: fx.Int32,
        num_seqs: fx.Int32,
        num_splits: fx.Int32,
        num_tokens_per_seq: fx.Int32,
        stream: fx.Stream = _DEFAULT_STREAM,
    ):
        mla_decode_reduce_kernel(
            split_data,
            split_lse,
            seq_lens,
            final_output,
            total_tokens,
            num_seqs,
            num_splits,
            num_tokens_per_seq,
            value_attrs=kernel_value_attrs,
        ).launch(
            grid=(H, total_tokens, 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )

    launch_mla_decode_reduce.compile_hints = dict(kernel_compile_hints)
    return launch_mla_decode_reduce
