# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
#
# Scattered-pointer fused MoE matvec for GGUF Q4_K_M expert weights.
#
# Designed for "streaming MoE" inference where expert weights live in K
# discontiguous device buffers (typically slots in a per-(layer, projection)
# LRU cache fed from NVMe SSD), rather than a single contiguous (E, N, K)
# tensor. The kernel takes an array of per-expert device pointers and a
# `remap[token, slot]` indirection that maps each (token, slot) pair to the
# index of its assigned expert pointer.
#
# Q4_K_M block layout (144 bytes per QK_K=256 elements):
#   bytes  0.. 3 : dm     = (half d, half dmin) super-block scales
#   bytes  4..15 : scales[12] -- 8 packed (sc, m) 6-bit pairs
#   bytes 16..143: qs[128]    -- 256 4-bit nibbles
#
# Sub-block (j in 0..7) decode:
#   j  < 4: sc[j] = raw[j]     & 0x3F        m[j] = raw[j+4] & 0x3F
#   j >= 4: sc[j] = (raw[j+4] & 0x0F) | ((raw[j-4] >> 6) << 4)
#           m[j]  = (raw[j+4] >> 4)   | ((raw[j  ] >> 6) << 4)
#
# Element layout within each block (e in 0..255):
#   sub-block s = e >> 5 covers elements [32*s .. 32*s+31].
#   For s = 2*il (low half): nibble = qs[32*il + (e & 31)] & 0x0F
#   For s = 2*il+1 (high half): nibble = qs[32*il + (e & 31)] >> 4
#   x[e] = (dall * sc[s]) * nibble - (dmin * m[s])
#
# Each program tiles BLOCK_SIZE_N output rows for one (token, slot). The
# inner block-loop unrolls the 8 sub-blocks (4 il-groups, low+high nibble
# each), avoiding both 2D gather operations and runtime nibble-position
# arithmetic.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


_moe_q4k_streaming_kernel_repr = make_kernel_repr(
    "_moe_q4k_streaming_kernel",
    [
        "QK_K",
        "BLOCK_BYTES",
        "BLOCK_SIZE_N",
    ],
)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 4}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 8}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_N": 32}, num_warps=4, num_stages=2),
    ],
    key=["n_dim_in", "n_dim_out"],
)
@triton.jit(repr=_moe_q4k_streaming_kernel_repr)
def _moe_q4k_streaming_kernel(
    # Pointers
    a_ptr,                  # *fp32 [n_tokens, n_dim_in]
    expert_ptrs_ptr,        # *uint64 [n_unique_experts]  (absolute device addresses)
    remap_ptr,              # *int32 [n_tokens, n_used_per_token]
    c_ptr,                  # *fp32 [n_tokens, n_used_per_token, n_dim_out]
    # Dimensions
    n_dim_in,
    n_dim_out,
    n_used_per_token,
    # Strides in elements
    stride_a_token,
    stride_c_token,
    stride_c_slot,
    # Compile-time meta
    QK_K: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """One program tiles BLOCK_SIZE_N output rows for one (token, slot).

    Grid: (cdiv(n_dim_out, BLOCK_SIZE_N), n_used_per_token, n_tokens)
    """
    pid_tile = tl.program_id(0)
    pid_slot = tl.program_id(1)
    pid_token = tl.program_id(2)

    row_start = pid_tile * BLOCK_SIZE_N
    row_offs = row_start + tl.arange(0, BLOCK_SIZE_N)
    row_mask = row_offs < n_dim_out
    safe_row = tl.where(row_mask, row_offs, 0)

    slot_idx = tl.load(remap_ptr + pid_token * n_used_per_token + pid_slot)
    expert_addr = tl.load(expert_ptrs_ptr + slot_idx)
    expert_base = tl.cast(expert_addr, tl.pointer_type(tl.uint8))

    n_blocks = n_dim_in // QK_K
    bytes_per_row = n_blocks * BLOCK_BYTES
    row_byte_base = expert_base + safe_row * bytes_per_row  # (BLOCK_SIZE_N,)

    x_token_base = a_ptr + pid_token * stride_a_token

    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    qs_offs = tl.arange(0, 32)

    for bi in range(0, n_blocks):
        block_base = row_byte_base + bi * BLOCK_BYTES  # (BLOCK_SIZE_N,)

        # dm: 2 fp16 per row -> dall, dmin (both BLOCK_SIZE_N,)
        dm_ptr = tl.cast(block_base, tl.pointer_type(tl.float16))
        dall = tl.load(dm_ptr + 0).to(tl.float32)
        dmin = tl.load(dm_ptr + 1).to(tl.float32)

        # 12 scale bytes -> 16 6-bit values (8 sc, 8 m)
        b0 = tl.load(block_base + 4 + 0).to(tl.int32)
        b1 = tl.load(block_base + 4 + 1).to(tl.int32)
        b2 = tl.load(block_base + 4 + 2).to(tl.int32)
        b3 = tl.load(block_base + 4 + 3).to(tl.int32)
        b4 = tl.load(block_base + 4 + 4).to(tl.int32)
        b5 = tl.load(block_base + 4 + 5).to(tl.int32)
        b6 = tl.load(block_base + 4 + 6).to(tl.int32)
        b7 = tl.load(block_base + 4 + 7).to(tl.int32)
        b8 = tl.load(block_base + 4 + 8).to(tl.int32)
        b9 = tl.load(block_base + 4 + 9).to(tl.int32)
        b10 = tl.load(block_base + 4 + 10).to(tl.int32)
        b11 = tl.load(block_base + 4 + 11).to(tl.int32)

        sc0 = b0 & 0x3F
        sc1 = b1 & 0x3F
        sc2 = b2 & 0x3F
        sc3 = b3 & 0x3F
        m0 = b4 & 0x3F
        m1 = b5 & 0x3F
        m2 = b6 & 0x3F
        m3 = b7 & 0x3F
        sc4 = (b8 & 0x0F) | ((b0 >> 6) << 4)
        sc5 = (b9 & 0x0F) | ((b1 >> 6) << 4)
        sc6 = (b10 & 0x0F) | ((b2 >> 6) << 4)
        sc7 = (b11 & 0x0F) | ((b3 >> 6) << 4)
        m4 = (b8 >> 4) | ((b4 >> 6) << 4)
        m5 = (b9 >> 4) | ((b5 >> 6) << 4)
        m6 = (b10 >> 4) | ((b6 >> 6) << 4)
        m7 = (b11 >> 4) | ((b7 >> 6) << 4)

        sc_f0 = dall * sc0.to(tl.float32);  m_f0 = dmin * m0.to(tl.float32)
        sc_f1 = dall * sc1.to(tl.float32);  m_f1 = dmin * m1.to(tl.float32)
        sc_f2 = dall * sc2.to(tl.float32);  m_f2 = dmin * m2.to(tl.float32)
        sc_f3 = dall * sc3.to(tl.float32);  m_f3 = dmin * m3.to(tl.float32)
        sc_f4 = dall * sc4.to(tl.float32);  m_f4 = dmin * m4.to(tl.float32)
        sc_f5 = dall * sc5.to(tl.float32);  m_f5 = dmin * m5.to(tl.float32)
        sc_f6 = dall * sc6.to(tl.float32);  m_f6 = dmin * m6.to(tl.float32)
        sc_f7 = dall * sc7.to(tl.float32);  m_f7 = dmin * m7.to(tl.float32)

        # 4 contiguous 32-byte qs chunks; each yields two sub-blocks.
        chunk0 = tl.load(block_base[:, None] + 16 + 0 + qs_offs[None, :]).to(tl.int32)
        x_sb0 = tl.load(x_token_base + bi * QK_K + 0 + qs_offs)
        x_sb1 = tl.load(x_token_base + bi * QK_K + 32 + qs_offs)
        nib_lo0 = chunk0 & 0x0F
        nib_hi0 = chunk0 >> 4
        accumulator += tl.sum(
            (sc_f0[:, None] * nib_lo0.to(tl.float32) - m_f0[:, None]) * x_sb0[None, :],
            axis=1,
        )
        accumulator += tl.sum(
            (sc_f1[:, None] * nib_hi0.to(tl.float32) - m_f1[:, None]) * x_sb1[None, :],
            axis=1,
        )

        chunk1 = tl.load(block_base[:, None] + 16 + 32 + qs_offs[None, :]).to(tl.int32)
        x_sb2 = tl.load(x_token_base + bi * QK_K + 64 + qs_offs)
        x_sb3 = tl.load(x_token_base + bi * QK_K + 96 + qs_offs)
        nib_lo1 = chunk1 & 0x0F
        nib_hi1 = chunk1 >> 4
        accumulator += tl.sum(
            (sc_f2[:, None] * nib_lo1.to(tl.float32) - m_f2[:, None]) * x_sb2[None, :],
            axis=1,
        )
        accumulator += tl.sum(
            (sc_f3[:, None] * nib_hi1.to(tl.float32) - m_f3[:, None]) * x_sb3[None, :],
            axis=1,
        )

        chunk2 = tl.load(block_base[:, None] + 16 + 64 + qs_offs[None, :]).to(tl.int32)
        x_sb4 = tl.load(x_token_base + bi * QK_K + 128 + qs_offs)
        x_sb5 = tl.load(x_token_base + bi * QK_K + 160 + qs_offs)
        nib_lo2 = chunk2 & 0x0F
        nib_hi2 = chunk2 >> 4
        accumulator += tl.sum(
            (sc_f4[:, None] * nib_lo2.to(tl.float32) - m_f4[:, None]) * x_sb4[None, :],
            axis=1,
        )
        accumulator += tl.sum(
            (sc_f5[:, None] * nib_hi2.to(tl.float32) - m_f5[:, None]) * x_sb5[None, :],
            axis=1,
        )

        chunk3 = tl.load(block_base[:, None] + 16 + 96 + qs_offs[None, :]).to(tl.int32)
        x_sb6 = tl.load(x_token_base + bi * QK_K + 192 + qs_offs)
        x_sb7 = tl.load(x_token_base + bi * QK_K + 224 + qs_offs)
        nib_lo3 = chunk3 & 0x0F
        nib_hi3 = chunk3 >> 4
        accumulator += tl.sum(
            (sc_f6[:, None] * nib_lo3.to(tl.float32) - m_f6[:, None]) * x_sb6[None, :],
            axis=1,
        )
        accumulator += tl.sum(
            (sc_f7[:, None] * nib_hi3.to(tl.float32) - m_f7[:, None]) * x_sb7[None, :],
            axis=1,
        )

    accumulator = tl.where(row_mask, accumulator, 0.0)
    dst_base = (
        c_ptr
        + pid_token * stride_c_token
        + pid_slot * stride_c_slot
        + row_offs
    )
    tl.store(dst_base, accumulator, mask=row_mask)
