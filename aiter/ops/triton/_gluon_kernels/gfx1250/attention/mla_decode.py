# SPDX-License-Identifier: Apache-2.0
"""Register-minimal gfx1250 MLA decode specialization.

This kernel deliberately covers the common DeepSeek MLA decode shape and
falls back to the general production kernel for every other configuration.
It handles all context lengths in one kernel, including one to three pages.
"""

import torch
import triton.experimental.gluon.language as gl
import triton.language as tl
from triton.experimental import gluon

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.types import e4m3_dtype

KV_LORA_RANK = gl.constexpr(512)
QK_ROPE_HEAD_DIM = gl.constexpr(64)
BLOCK_SIZE = gl.constexpr(64)
BLOCK_M = gl.constexpr(128)
NUM_WARPS = 4
FP8_MIN_VALUE = torch.finfo(e4m3_dtype).min
FP8_MAX_VALUE = torch.finfo(e4m3_dtype).max
# Four TDM stages; LDS operands are consumed within their tile.
RING = gl.constexpr(4)
RCP_LN2 = gl.constexpr(1.4426950408889634)

QK_WMMA: gl.constexpr = gl.amd.AMDWMMALayout(
    version=3,
    transposed=True,
    warp_bases=[(1, 0), (2, 0)],
    reg_bases=[],
    instr_shape=[16, 16, 64],
)
PV_WMMA: gl.constexpr = gl.amd.AMDWMMALayout(
    version=3,
    transposed=True,
    warp_bases=[(0, 1), (0, 2)],
    reg_bases=[],
    instr_shape=[16, 16, 64],
)
Q_DOT: gl.constexpr = gl.DotOperandLayout(0, QK_WMMA, 32)
K_DOT: gl.constexpr = gl.DotOperandLayout(1, QK_WMMA, 32)
P_DOT: gl.constexpr = gl.DotOperandLayout(0, PV_WMMA, 32)
V_DOT: gl.constexpr = gl.DotOperandLayout(1, PV_WMMA, 32)

Q_LORA_SMEM: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
    [[KV_LORA_RANK.value, 8]],
    [BLOCK_M.value, KV_LORA_RANK.value],
    [1, 0],
)
Q_ROPE_SMEM: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
    [[QK_ROPE_HEAD_DIM.value, 8]],
    [BLOCK_M.value, QK_ROPE_HEAD_DIM.value],
    [1, 0],
)
KV_SMEM: gl.constexpr = gl.SwizzledSharedLayout(
    vec=1, per_phase=1, max_phase=1, order=[1, 0]
)
Q_LORA_LOAD: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 8],
    threads_per_warp=[1, 32],
    warps_per_cta=[NUM_WARPS, 1],
    order=[1, 0],
)
Q_ROPE_LOAD: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 8],
    threads_per_warp=[4, 8],
    warps_per_cta=[NUM_WARPS, 1],
    order=[1, 0],
)


@gluon.jit
def _unshuffle_lora(smem, buffer_id):
    return (
        smem.index(buffer_id)
        .reshape((1, BLOCK_SIZE // 16, KV_LORA_RANK // 32, 2, 16, 16))
        .permute((0, 1, 4, 2, 3, 5))
        .reshape((BLOCK_SIZE, KV_LORA_RANK))
    )


@gluon.jit
def _unshuffle_rope(smem, buffer_id):
    return (
        smem.index(buffer_id)
        .reshape((1, BLOCK_SIZE // 16, QK_ROPE_HEAD_DIM // 32, 2, 16, 16))
        .permute((0, 1, 4, 2, 3, 5))
        .reshape((BLOCK_SIZE, QK_ROPE_HEAD_DIM))
    )


@gluon.jit
def _load_page(desc, smem, row_idx, buffer_id):
    gl.amd.gfx1250.tdm.async_load(desc, [row_idx, 0], smem.index(buffer_id))


@gluon.jit
def _process_tile(
    q_lora,
    q_rope,
    kv_lora_smem,
    k_rope_smem,
    buffer_id,
    tile_idx,
    qk_factor,
    seq_len,
    L,
    M,
    acc,
    wait_rope: gl.constexpr,
    IS_LAST: gl.constexpr,
):
    S = gl.zeros([BLOCK_M, BLOCK_SIZE], dtype=tl.float32, layout=QK_WMMA)
    # RoPE follows LoRA in TDM issue order. Wait for both before issuing the
    # LDS reads, so their latency can overlap with independent QK matrix work.
    gl.amd.gfx1250.tdm.async_wait(wait_rope)
    k_lora = _unshuffle_lora(kv_lora_smem, buffer_id).permute((1, 0)).load(layout=K_DOT)
    k_rope = _unshuffle_rope(k_rope_smem, buffer_id).permute((1, 0)).load(layout=K_DOT)
    S = gl.amd.gfx1250.wmma(q_lora, k_lora, S)
    S = gl.amd.gfx1250.wmma(q_rope, k_rope, S) * qk_factor

    if IS_LAST:
        pos = tile_idx * BLOCK_SIZE + gl.arange(
            0, BLOCK_SIZE, layout=gl.SliceLayout(0, QK_WMMA)
        )
        S = gl.where(pos[None, :] < seq_len, S, float("-inf"))

    # Issue V's LDS read before the independent softmax arithmetic.
    v = _unshuffle_lora(kv_lora_smem, buffer_id).load(layout=V_DOT)
    m_ij = gl.maximum(M, gl.max(S, axis=1))
    p = gl.exp2(S - m_ij[:, None])
    alpha = gl.exp2(M - m_ij)
    M = m_ij
    L = L * alpha + gl.sum(p, 1)
    acc = acc * gl.convert_layout(alpha[:, None], layout=PV_WMMA)
    # Publish the entire P tile together. The implicit conversion reuses
    # 4 KiB for two 64-row groups, requiring an extra scratch-reuse barrier.
    # This 8 KiB layout preserves its 16-byte vectorization and bank mapping.
    P_SHARED: gl.constexpr = gl.SharedLinearLayout(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 16],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [0, 8],
            [0, 32],
            [64, 0],
        ]
    )
    p_smem = gl.allocate_shared_memory(v.dtype, (BLOCK_M, BLOCK_SIZE), P_SHARED)
    p_smem.store(p.to(v.dtype))
    p = p_smem.load(P_DOT)
    acc = gl.amd.gfx1250.wmma(p, v, acc)
    return L, M, acc


@gluon.jit
def _process_short(
    q_lora,
    q_rope,
    kv_lora_smem,
    k_rope_smem,
    qk_factor,
    seq_len,
    L,
    M,
    acc,
    N_TILES: gl.constexpr,
):
    for tile in gl.static_range(N_TILES):
        L, M, acc = _process_tile(
            q_lora,
            q_rope,
            kv_lora_smem,
            k_rope_smem,
            tile,
            tile,
            qk_factor,
            seq_len,
            L,
            M,
            acc,
            wait_rope=2 * (N_TILES - tile) - 2,
            IS_LAST=(tile == N_TILES - 1),
        )
    return L, M, acc


_kernel_repr = make_kernel_repr(
    "_mla_decode_fwd_kernel_specialized",
    ["num_kv_heads"],
)


@gluon.jit(repr=_kernel_repr)
def _mla_decode_fwd_kernel_specialized(
    output_ptr,
    query_ptr,
    kv_buffer_ptr,
    block_tables_ptr: tl.const,
    seq_lens_ptr,
    query_start_len_ptr,
    SCALE: gl.constexpr,
    q_scale_ptr,
    kv_scale_ptr,
    out_scale_ptr,
    num_kv_heads: gl.constexpr,
    block_tables_stride: gl.int32,
    query_stride_0: gl.int32,
    query_stride_1: gl.int32,
    output_stride_0: gl.int32,
    output_stride_1: gl.int32,
    stride_kv_buffer_1: gl.int32,
    num_blocks: gl.int32,
    TDM_STORE: gl.constexpr = False,
    FP8_MIN: gl.constexpr = FP8_MIN_VALUE,
    FP8_MAX: gl.constexpr = FP8_MAX_VALUE,
):
    seq_idx = gl.program_id(0)
    if num_kv_heads == 1:
        kv_head_idx = 0
    else:
        kv_head_idx = gl.program_id(1)
    seq_len = gl.load(seq_lens_ptr + seq_idx)
    # ALL_DECODE means every packed sequence contributes exactly one Q token.
    q_start_idx = seq_idx
    num_tiles = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    qk_factor: gl.float32 = SCALE * RCP_LN2
    if q_scale_ptr is not None:
        qk_factor *= gl.load(q_scale_ptr)
    out_factor: gl.float32 = 1.0
    if kv_scale_ptr is not None:
        kv_scale = gl.load(kv_scale_ptr)
        qk_factor *= kv_scale
        out_factor = kv_scale
    if out_scale_ptr is not None:
        out_factor /= gl.load(out_scale_ptr)

    offs_m_l = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_LORA_LOAD))
    offs_d_l = gl.arange(0, KV_LORA_RANK, layout=gl.SliceLayout(0, Q_LORA_LOAD))
    offs_m_r = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_ROPE_LOAD))
    offs_d_r = gl.arange(0, QK_ROPE_HEAD_DIM, layout=gl.SliceLayout(0, Q_ROPE_LOAD))
    head_base = kv_head_idx * BLOCK_M
    q_base = query_ptr + q_start_idx * query_stride_0 + head_base * query_stride_1

    q_lora_smem = gl.allocate_shared_memory(
        query_ptr.type.element_ty, [BLOCK_M, KV_LORA_RANK], Q_LORA_SMEM
    )
    q_lora_smem.store(
        gl.load(q_base + offs_m_l[:, None] * query_stride_1 + offs_d_l[None, :])
    )
    q_lora = q_lora_smem.load(layout=Q_DOT)

    q_rope_smem = gl.allocate_shared_memory(
        query_ptr.type.element_ty, [BLOCK_M, QK_ROPE_HEAD_DIM], Q_ROPE_SMEM
    )
    q_rope_smem.store(
        gl.load(
            q_base
            + offs_m_r[:, None] * query_stride_1
            + (KV_LORA_RANK + offs_d_r)[None, :]
        )
    )
    q_rope = q_rope_smem.load(layout=Q_DOT)

    kv_lora_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=kv_buffer_ptr,
        shape=(num_blocks * num_kv_heads, BLOCK_SIZE * KV_LORA_RANK),
        strides=(stride_kv_buffer_1, 1),
        block_shape=(gl.constexpr(1), BLOCK_SIZE * KV_LORA_RANK),
        layout=KV_SMEM,
    )
    k_rope_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=kv_buffer_ptr + BLOCK_SIZE * KV_LORA_RANK,
        shape=(num_blocks * num_kv_heads, BLOCK_SIZE * QK_ROPE_HEAD_DIM),
        strides=(stride_kv_buffer_1, 1),
        block_shape=(gl.constexpr(1), BLOCK_SIZE * QK_ROPE_HEAD_DIM),
        layout=KV_SMEM,
    )
    kv_lora_smem = gl.allocate_shared_memory(
        kv_lora_desc.dtype, [RING] + kv_lora_desc.block_shape, KV_SMEM
    )
    k_rope_smem = gl.allocate_shared_memory(
        k_rope_desc.dtype, [RING] + k_rope_desc.block_shape, KV_SMEM
    )

    table = block_tables_ptr + seq_idx * block_tables_stride
    for prime in gl.static_range(3):
        if num_tiles > prime:
            page = gl.load(table + prime)
            if num_kv_heads == 1:
                row = page
            else:
                row = page * num_kv_heads + kv_head_idx
            _load_page(kv_lora_desc, kv_lora_smem, row, prime)
            _load_page(k_rope_desc, k_rope_smem, row, prime)

    M = gl.full([BLOCK_M], float("-inf"), gl.float32, gl.SliceLayout(1, QK_WMMA))
    L = gl.full([BLOCK_M], 1.0, gl.float32, gl.SliceLayout(1, QK_WMMA))
    acc = gl.zeros([BLOCK_M, KV_LORA_RANK], dtype=gl.float32, layout=PV_WMMA)

    if num_tiles > 3:
        slot: gl.int32 = 0
        page = gl.load(table + 3)
        for tile in range(num_tiles - 3):
            fill_slot = (slot + 3) % RING
            if num_kv_heads == 1:
                row = page
            else:
                row = page * num_kv_heads + kv_head_idx
            _load_page(kv_lora_desc, kv_lora_smem, row, fill_slot)
            _load_page(k_rope_desc, k_rope_smem, row, fill_slot)
            page = gl.load(table + gl.minimum(tile + 4, num_tiles - 1))
            L, M, acc = _process_tile(
                q_lora,
                q_rope,
                kv_lora_smem,
                k_rope_smem,
                slot,
                tile,
                qk_factor,
                seq_len,
                L,
                M,
                acc,
                wait_rope=6,
                IS_LAST=False,
            )
            slot = (slot + 1) % RING

        for k in gl.static_range(3):
            L, M, acc = _process_tile(
                q_lora,
                q_rope,
                kv_lora_smem,
                k_rope_smem,
                slot,
                num_tiles - 3 + k,
                qk_factor,
                seq_len,
                L,
                M,
                acc,
                wait_rope=2 * (3 - k) - 2,
                IS_LAST=(k == 2),
            )
            slot = (slot + 1) % RING
    elif num_tiles > 2:
        L, M, acc = _process_short(
            q_lora,
            q_rope,
            kv_lora_smem,
            k_rope_smem,
            qk_factor,
            seq_len,
            L,
            M,
            acc,
            N_TILES=3,
        )
    elif num_tiles > 1:
        L, M, acc = _process_short(
            q_lora,
            q_rope,
            kv_lora_smem,
            k_rope_smem,
            qk_factor,
            seq_len,
            L,
            M,
            acc,
            N_TILES=2,
        )
    else:
        L, M, acc = _process_short(
            q_lora,
            q_rope,
            kv_lora_smem,
            k_rope_smem,
            qk_factor,
            seq_len,
            L,
            M,
            acc,
            N_TILES=1,
        )

    output_base = (
        output_ptr + q_start_idx * output_stride_0 + head_base * output_stride_1
    )
    factor = gl.convert_layout(1.0 / L[:, None], layout=PV_WMMA) * out_factor
    if TDM_STORE:
        # Stage four independent slices, starting each TDM store before the
        # next slice is written. Dead attention buffers supply the LDS space.
        STORE_N: gl.constexpr = 128
        O_LAYOUT: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[STORE_N, 8]],
            [BLOCK_M, STORE_N],
            [1, 0],
        )
        o_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=output_base,
            shape=(BLOCK_M, KV_LORA_RANK),
            strides=(output_stride_1, 1),
            block_shape=(BLOCK_M, STORE_N),
            layout=O_LAYOUT,
        )
        output_buffers = ()
        for n in gl.static_range(0, KV_LORA_RANK, STORE_N):
            part = gl.amd.slice(acc, [BLOCK_M, STORE_N], [0, n])
            result = (part * factor).to(output_ptr.type.element_ty)
            o_smem = gl.allocate_shared_memory(
                output_ptr.type.element_ty,
                [BLOCK_M, STORE_N],
                O_LAYOUT,
            )
            output_buffers += (o_smem,)
            o_smem.store(result)
            gl.amd.gfx1250.tdm.async_store(o_desc, [0, n], o_smem)
        gl.amd.gfx1250.tdm.async_wait(0)
        # All sources must remain disjoint until every asynchronous store ends.
        for i in gl.static_range(len(output_buffers)):
            output_buffers[i]._keep_alive()
    else:
        acc *= factor
        if output_ptr.type.element_ty.is_fp8():
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
        offs_m_o = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, PV_WMMA))
        offs_d_o = gl.arange(0, KV_LORA_RANK, layout=gl.SliceLayout(0, PV_WMMA))
        gl.store(
            output_base + offs_m_o[:, None] * output_stride_1 + offs_d_o[None, :],
            acc.to(output_ptr.type.element_ty),
        )
