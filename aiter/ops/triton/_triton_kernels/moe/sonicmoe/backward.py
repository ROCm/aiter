# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_db2_and_ds_repr = make_kernel_repr(
    "sonicmoe_db2_and_ds",
    ["H", "E", "OLD_DS_PARTIAL_N", "BLOCK_H", "BLOCK_TK", "BLOCK_OLD_DS_PARTIAL_N"],
)
_db1_repr = make_kernel_repr(
    "sonicmoe_db1", ["I", "E", "BLOCK_I", "BLOCK_TK", "CONCAT_LAYOUT"]
)


@triton.jit(repr=_db2_and_ds_repr)
def db2_and_ds_kernel(
    dout_ptr,
    s_ptr,
    new_ds_partial_ptr,
    old_ds_partial_ptr,
    b2_ptr,
    db2_ptr,
    x_gather_idx_ptr,
    s_scatter_idx_ptr,
    expert_offset_ptr,
    H: tl.constexpr,
    E: tl.constexpr,
    OLD_DS_PARTIAL_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    BLOCK_OLD_DS_PARTIAL_N: tl.constexpr,
):
    Eidx = tl.program_id(0)
    Hidx = tl.program_id(1)
    NUM_H_BLOCKS: tl.constexpr = tl.num_programs(1)

    h_offsets = Hidx * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    E_count_start = tl.load(expert_offset_ptr + Eidx)
    E_count_end = tl.load(expert_offset_ptr + Eidx + 1)
    n_tokens = E_count_end - E_count_start

    b2 = tl.load(b2_ptr + Eidx * H + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    db2_acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for block_start in tl.range(0, n_tokens, BLOCK_TK):
        tk_offsets = block_start + tl.arange(0, BLOCK_TK)
        tk_mask = tk_offsets < n_tokens
        tk_grouped = E_count_start + tk_offsets

        token_indices = tl.load(
            x_gather_idx_ptr + tk_grouped, mask=tk_mask, other=0
        ).to(tl.int64)
        scatter_indices = tl.load(
            s_scatter_idx_ptr + tk_grouped, mask=tk_mask, other=0
        ).to(tl.int64)
        s = tl.load(s_ptr + scatter_indices, mask=tk_mask, other=0.0).to(tl.float32)

        dout_offsets = token_indices[:, None] * H + h_offsets[None, :]
        dout_mask = tk_mask[:, None] & h_mask[None, :]
        dout = tl.load(dout_ptr + dout_offsets, mask=dout_mask, other=0.0).to(
            tl.float32
        )

        db2_acc += tl.sum(dout * s[:, None], axis=0)

        ds_partial = tl.sum(dout * b2[None, :], axis=1)

        if Hidx == 0:
            n_offsets = tl.arange(0, BLOCK_OLD_DS_PARTIAL_N)
            old_ds_partial_offsets = (
                scatter_indices[:, None] * OLD_DS_PARTIAL_N + n_offsets[None, :]
            )
            old_ds_partial_mask = tk_mask[:, None] & (
                n_offsets[None, :] < OLD_DS_PARTIAL_N
            )
            old_ds_partial_vals = tl.load(
                old_ds_partial_ptr + old_ds_partial_offsets,
                mask=old_ds_partial_mask,
                other=0.0,
            ).to(tl.float32)
            ds_partial += tl.sum(old_ds_partial_vals, axis=1)

        tl.store(
            new_ds_partial_ptr + scatter_indices * NUM_H_BLOCKS + Hidx,
            ds_partial,
            mask=tk_mask,
        )

    tl.store(db2_ptr + Eidx * H + h_offsets, db2_acc, mask=h_mask)


@triton.jit(repr=_db1_repr)
def db1_kernel(
    dh_ptr,
    db1_ptr,
    expert_offset_ptr,
    I: tl.constexpr,
    E: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    CONCAT_LAYOUT: tl.constexpr = False,
):
    Eidx = tl.program_id(0)

    E_count_start = tl.load(expert_offset_ptr + Eidx).to(tl.int64)
    E_count_end = tl.load(expert_offset_ptr + Eidx + 1).to(tl.int64)
    n_tokens = E_count_end - E_count_start

    NUM_I_BLOCKS: tl.constexpr = triton.cdiv(I, BLOCK_I)
    I_HALF: tl.constexpr = I // 2
    for Iidx in tl.static_range(0, NUM_I_BLOCKS, 1):
        i_offsets = Iidx * BLOCK_I + tl.arange(0, BLOCK_I)
        i_mask = i_offsets < I

        db1_acc = tl.zeros([BLOCK_I], dtype=tl.float32)

        for block_start in tl.range(0, n_tokens, BLOCK_TK):
            tk_offsets = block_start + tl.arange(0, BLOCK_TK)
            tk_mask = tk_offsets < n_tokens
            tk_grouped = E_count_start + tk_offsets

            dz_offsets = tk_grouped[:, None] * I + i_offsets[None, :]
            dz_mask = tk_mask[:, None] & i_mask[None, :]
            dz = tl.load(dh_ptr + dz_offsets, mask=dz_mask, other=0.0).to(tl.float32)
            db1_acc += tl.sum(dz, axis=0)

        if CONCAT_LAYOUT:
            out_offsets = i_offsets // 2 + (i_offsets % 2) * I_HALF
        else:
            out_offsets = i_offsets
        db1_offsets = Eidx.to(tl.int64) * I + out_offsets
        tl.store(db1_ptr + db1_offsets, db1_acc, mask=i_mask)
