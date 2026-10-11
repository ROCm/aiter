# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# Sliding-window paged decode over the shuffled fp8 KV cache. The asm pa_fwd
# decode path takes no window arg, so SWA layers can't use the shuffled cache;
# this reads the same layout and masks the window.
#
# Shuffled layout (x = 16 // elem_size, = 16 for fp8):
#   K: [num_blocks, num_kv_heads, head_size // x, block_size, x]
#   V: [num_blocks, num_kv_heads, block_size // x, head_size, x]
#   K[d, slot] -> (d // x) * block_size * x + slot * x + (d % x)
#   V[slot, d] -> (slot // x) * head_size * x + d * x + (slot % x)
# A16W8: q is 16-bit, fp8 KV dequantized (val * scale) before the dot.


@triton.jit
def _pa_decode_shuffle_swa_kernel(
    out_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    blk_tbl_ptr,
    seq_lens_ptr,
    k_scale_ptr,
    v_scale_ptr,
    scale,
    sliding_window,
    q_s0,
    q_s1,
    q_s2,
    o_s0,
    o_s1,
    o_s2,
    bt_s0,
    ks_s0,
    ks_s1,
    ks_s2,
    k_blk_stride,
    k_head_stride,
    v_blk_stride,
    v_head_stride,
    HEAD: tl.constexpr,
    HEAD_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
    X: tl.constexpr,
    GRP: tl.constexpr,
    GRP_POW2: tl.constexpr,
    SCALE_SCALAR: tl.constexpr,
):
    seq = tl.program_id(0)
    kvh = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + seq)
    win_start = 0
    if sliding_window > 0:
        ws = seq_len - sliding_window
        win_start = tl.where(ws > 0, ws, 0)

    grp = tl.arange(0, GRP_POW2)
    d = tl.arange(0, HEAD_POW2)
    q_head = kvh * GRP + grp
    q_mask = (grp[:, None] < GRP) & (d[None, :] < HEAD)
    q_ptrs = q_ptr + seq * q_s0 + q_head[:, None] * q_s1 + d[None, :] * q_s2
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    m_i = tl.full([GRP_POW2], -float("inf"), tl.float32)
    l_i = tl.zeros([GRP_POW2], tl.float32)
    acc = tl.zeros([GRP_POW2, HEAD_POW2], tl.float32)

    s = tl.arange(0, BLOCK)
    n_blk = tl.cdiv(seq_len, BLOCK)
    start_blk = win_start // BLOCK

    if SCALE_SCALAR:
        ks_scalar = tl.load(k_scale_ptr)
        vs_scalar = tl.load(v_scale_ptr)

    for b in range(start_blk, n_blk):
        block_id = tl.load(blk_tbl_ptr + seq * bt_s0 + b).to(tl.int64)
        pos = b * BLOCK + s
        valid = (pos < seq_len) & (pos >= win_start)

        k_base = block_id * k_blk_stride + kvh * k_head_stride
        v_base = block_id * v_blk_stride + kvh * v_head_stride
        ld_mask = (s[:, None] < BLOCK) & (d[None, :] < HEAD)

        k_off = (
            k_base + (d[None, :] // X) * (BLOCK * X) + s[:, None] * X + (d[None, :] % X)
        )
        v_off = (
            v_base + (s[:, None] // X) * (HEAD * X) + d[None, :] * X + (s[:, None] % X)
        )

        k = tl.load(k_cache_ptr + k_off, mask=ld_mask, other=0.0).to(tl.float32)
        v = tl.load(v_cache_ptr + v_off, mask=ld_mask, other=0.0).to(tl.float32)

        if SCALE_SCALAR:
            k = k * ks_scalar
            v = v * vs_scalar
        else:
            sc_off = block_id * ks_s0 + kvh * ks_s1 + s * ks_s2
            ks = tl.load(k_scale_ptr + sc_off, mask=s < BLOCK, other=0.0)
            vs = tl.load(v_scale_ptr + sc_off, mask=s < BLOCK, other=0.0)
            k = k * ks[:, None]
            v = v * vs[:, None]

        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(valid[None, :], scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
        m_i = m_new

    out = acc / l_i[:, None]
    o_ptrs = out_ptr + seq * o_s0 + q_head[:, None] * o_s1 + d[None, :] * o_s2
    tl.store(o_ptrs, out.to(out_ptr.dtype.element_ty), mask=q_mask)


def paged_attention_decode_shuffle_swa(
    out,  # [num_seqs, num_q_heads, head_size]
    query,  # [num_seqs, num_q_heads, head_size]
    key_cache,  # [num_blocks, num_kv_heads, head_size // x, block_size, x]
    value_cache,  # [num_blocks, num_kv_heads, block_size // x, head_size, x]
    block_tables,  # [num_seqs, max_num_blocks]
    seq_lens,  # [num_seqs]
    scale,  # softmax scale
    k_scale,  # scalar or [num_blocks, num_kv_heads, block_size]
    v_scale,
    sliding_window,  # number of keys attended (<= 0 means full causal)
):
    """Sliding-window decode over the shuffled fp8 KV layout. One program per
    (sequence, kv head); decode only (one query token per sequence)."""
    num_seqs, num_q_heads, head = query.shape
    num_kv_heads = key_cache.shape[1]
    block_size = key_cache.shape[3]
    x = key_cache.shape[4]
    grp = num_q_heads // num_kv_heads
    grp_pow2 = max(16, triton.next_power_of_2(grp))
    head_pow2 = triton.next_power_of_2(head)

    _LOGGER.info(
        f"PA_DECODE_SHUFFLE_SWA: q={tuple(query.shape)} "
        f"k_cache={tuple(key_cache.shape)} window={sliding_window}"
    )

    scale_scalar = k_scale.numel() == 1
    if scale_scalar:
        ks_s0 = ks_s1 = ks_s2 = 0
    else:
        ks_s0, ks_s1, ks_s2 = k_scale.stride()

    grid = (num_seqs, num_kv_heads)
    _pa_decode_shuffle_swa_kernel[grid](
        out,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        scale,
        sliding_window,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        block_tables.stride(0),
        ks_s0,
        ks_s1,
        ks_s2,
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        HEAD=head,
        HEAD_POW2=head_pow2,
        BLOCK=block_size,
        X=x,
        GRP=grp,
        GRP_POW2=grp_pow2,
        SCALE_SCALAR=scale_scalar,
    )
    return out
