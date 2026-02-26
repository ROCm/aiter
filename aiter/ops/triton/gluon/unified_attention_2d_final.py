from typing import Optional, Union

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.language.core import _aggregate as aggregate
import pytest
from aiter.ops.triton.utils._triton import arch_info
import os
PRINT_IRS = os.environ.get("PRINT_IRS", "0") == "1"
@aggregate
class AsyncKVLoaderConfig:
    """Configuration for asynchronous KV loader."""
    blocked_k: gl.constexpr
    blocked_v: gl.constexpr
    shared_k_layout: gl.constexpr
    shared_v_layout: gl.constexpr
    USE_LOAD_BUFFER_OP: gl.constexpr
    KV_CACHE_MODIFIER: gl.constexpr

    k_reg_layout: gl.constexpr
    v_reg_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, cfg):
        # Blocked layouts for global-to-shared memory loads
        HEAD_SIZE_DIV = cfg.HEAD_SIZE // 8
        #gl.static_assert(WARP_SIZE % HEAD_SIZE_DIV == 0, "WARP_SIZE must be divisible by HEAD_SIZE_DIV")
        self.blocked_v = gl.constexpr(gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[cfg.WARP_SIZE // HEAD_SIZE_DIV, HEAD_SIZE_DIV],
            warps_per_cta=[cfg.NUM_WARPS, 1],
            order=[1, 0],
        ))
        self.blocked_k = gl.constexpr(gl.BlockedLayout(
            size_per_thread=[8, 1],
            threads_per_warp=[HEAD_SIZE_DIV, cfg.WARP_SIZE // HEAD_SIZE_DIV],
            warps_per_cta=[1, cfg.NUM_WARPS],
            order=[0, 1],
        ))

        # Swizzled shared memory layouts for K and V
        self.shared_k_layout = gl.constexpr(
            gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[0, 1]))
        self.shared_v_layout = gl.constexpr(
            gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0]))

        self.KV_CACHE_MODIFIER = cfg.KV_CACHE_MODIFIER
        self.USE_LOAD_BUFFER_OP = cfg.USE_LOAD_BUFFER_OP

        self.k_reg_layout = gl.constexpr(cfg.k_layout)
        self.v_reg_layout = gl.constexpr(cfg.v_layout)

@aggregate
class AsyncKVLoader:
    kv_cfg: AsyncKVLoaderConfig
    key_cache_ptr: gl.tensor
    value_cache_ptr: gl.tensor
    k_shared: gl.shared_memory_descriptor
    v_shared: gl.shared_memory_descriptor
    k_base_offset: gl.tensor
    v_base_offset: gl.tensor
    @gluon.constexpr_function
    def __init__(self, kv_cfg, key_cache_ptr, value_cache_ptr, k_shared, v_shared, k_base_offset, v_base_offset):
        self.kv_cfg = kv_cfg
        self.key_cache_ptr = key_cache_ptr
        self.value_cache_ptr = value_cache_ptr
        self.k_shared = k_shared
        self.v_shared = v_shared
        self.k_base_offset = k_base_offset
        self.v_base_offset = v_base_offset 
    @gluon.jit
    def initialize(cfg, key_cache_ptr, value_cache_ptr,
                 kv_head_idx, num_blocks,
                 stride_k_cache_1, stride_k_cache_2, stride_k_cache_3,
                 stride_v_cache_1, stride_v_cache_2, stride_v_cache_3):
        kv_cfg = AsyncKVLoaderConfig(cfg)
        k_shared = gl.allocate_shared_memory(
            key_cache_ptr.type.element_ty, [2, cfg.HEAD_SIZE, cfg.BLOCK_SIZE], layout=kv_cfg.shared_k_layout)
        v_shared = gl.allocate_shared_memory(
            value_cache_ptr.type.element_ty, [2, cfg.BLOCK_SIZE, cfg.HEAD_SIZE], layout=kv_cfg.shared_v_layout)
        
        # Precompute KV load offsets (constant across tiles) 
        offs_d_k = gl.arange(0, cfg.HEAD_SIZE, layout=gl.SliceLayout(1, kv_cfg.blocked_k))[:, None]
        offs_n_k = gl.arange(0, cfg.BLOCK_SIZE, layout=gl.SliceLayout(0, kv_cfg.blocked_k))[None, :]
        k_base_offset = (kv_head_idx * stride_k_cache_2
                         + offs_d_k * stride_k_cache_3
                         + offs_n_k * stride_k_cache_1)
        
        offs_d_v = gl.arange(0, cfg.HEAD_SIZE, layout=gl.SliceLayout(0, kv_cfg.blocked_v))[None, :]
        offs_n_v = gl.arange(0, cfg.BLOCK_SIZE, layout=gl.SliceLayout(1, kv_cfg.blocked_v))[:, None]
        v_base_offset = (kv_head_idx * stride_v_cache_2
                         + offs_d_v * stride_v_cache_3
                         + offs_n_v * stride_v_cache_1)
        
        return AsyncKVLoader(kv_cfg, key_cache_ptr, value_cache_ptr,
         k_shared, v_shared, k_base_offset, v_base_offset)
    @gluon.jit
    def load_k_to_shared(self, k_offset, buffer_id):
        # Async copy K tile from global to shared memory
        if self.kv_cfg.USE_LOAD_BUFFER_OP:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                self.k_shared.index(buffer_id), self.key_cache_ptr, self.k_base_offset + k_offset, cache_modifier=self.kv_cfg.KV_CACHE_MODIFIER)
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(
                self.k_shared.index(buffer_id), self.key_cache_ptr + self.k_base_offset + k_offset, cache_modifier=self.kv_cfg.KV_CACHE_MODIFIER)
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def load_v_to_shared(self, v_offset, buffer_id):
        # Async copy V tile from global to shared memory
        if self.kv_cfg.USE_LOAD_BUFFER_OP:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                self.v_shared.index(buffer_id), self.value_cache_ptr, self.v_base_offset + v_offset, cache_modifier=self.kv_cfg.KV_CACHE_MODIFIER)
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(
                self.v_shared.index(buffer_id), self.value_cache_ptr + self.v_base_offset + v_offset, cache_modifier=self.kv_cfg.KV_CACHE_MODIFIER)
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def load_k_from_shared(self, wait_count, buffer_id):
        # Wait for async K copy and load from shared memory
        gl.amd.cdna4.async_copy.wait_group(wait_count)
        return gl.amd.cdna4.async_copy.load_shared_relaxed(
            self.k_shared.index(buffer_id), self.kv_cfg.k_reg_layout)

    @gluon.jit
    def load_v_from_shared(self, wait_count, buffer_id):
        # Wait for async V copy and load from shared memory
        gl.amd.cdna4.async_copy.wait_group(wait_count)
        return gl.amd.cdna4.async_copy.load_shared_relaxed(
            self.v_shared.index(buffer_id), self.kv_cfg.v_reg_layout)

@aggregate
class TDMKVLoaderConfig:
    """Configuration for TDM KV loader."""
    shared_k_layout: gl.constexpr
    shared_v_layout: gl.constexpr
    USE_LOAD_BUFFER_OP: gl.constexpr
    KV_CACHE_MODIFIER: gl.constexpr

    k_reg_layout: gl.constexpr
    v_reg_layout: gl.constexpr
    BLOCK_SIZE: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, cfg):
        # Swizzled shared memory layouts for K and V
        # self.shared_k_layout = gl.constexpr(
        #     gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[0, 1]))
        # self.shared_v_layout = gl.constexpr(
        #     gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0]))
        self.shared_k_layout = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for([[cfg.HEAD_SIZE, 8]], [cfg.BLOCK_SIZE, cfg.HEAD_SIZE], [1, 0]))
        self.shared_v_layout = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for([[cfg.HEAD_SIZE, 16]], [cfg.BLOCK_SIZE, cfg.HEAD_SIZE], [1, 0]))


        self.KV_CACHE_MODIFIER = cfg.KV_CACHE_MODIFIER
        self.USE_LOAD_BUFFER_OP = cfg.USE_LOAD_BUFFER_OP

        self.k_reg_layout = gl.constexpr(cfg.k_layout)
        self.v_reg_layout = gl.constexpr(cfg.v_layout)
        self.BLOCK_SIZE = gl.constexpr(cfg.BLOCK_SIZE)
@aggregate
class TDMKVLoader:
    kv_cfg: TDMKVLoaderConfig
    k_shared: gl.shared_memory_descriptor
    v_shared: gl.shared_memory_descriptor
    k_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    v_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    kv_head_idx: gl.tensor
    stride_k_cache_2: gl.tensor
    stride_v_cache_2: gl.tensor

    @gluon.constexpr_function
    def __init__(self, kv_cfg, k_shared, v_shared, k_desc, v_desc, kv_head_idx, stride_k_cache_2, stride_v_cache_2):
        self.kv_cfg = kv_cfg
        self.k_shared = k_shared
        self.v_shared = v_shared
        self.k_desc = k_desc
        self.v_desc = v_desc
        self.kv_head_idx = kv_head_idx
        self.stride_k_cache_2 = stride_k_cache_2
        self.stride_v_cache_2 = stride_v_cache_2
    
    @gluon.jit
    def initialize(cfg, key_cache_ptr, value_cache_ptr,
                 kv_head_idx, num_blocks,
                 stride_k_cache_1, stride_k_cache_2, stride_k_cache_3,
                 stride_v_cache_1, stride_v_cache_2, stride_v_cache_3):
        kv_cfg = TDMKVLoaderConfig(cfg)
        k_shared = gl.allocate_shared_memory(
            key_cache_ptr.type.element_ty, [2, cfg.BLOCK_SIZE, cfg.HEAD_SIZE], layout=kv_cfg.shared_k_layout)
        v_shared = gl.allocate_shared_memory(
            value_cache_ptr.type.element_ty, [2, cfg.BLOCK_SIZE, cfg.HEAD_SIZE], layout=kv_cfg.shared_v_layout)
        
        k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=key_cache_ptr,
                shape=(num_blocks * cfg.BLOCK_SIZE, cfg.NUM_KV_HEADS * cfg.HEAD_SIZE),
                strides=(stride_k_cache_1, stride_k_cache_3),
                block_shape=(cfg.BLOCK_SIZE, cfg.HEAD_SIZE),
                layout=kv_cfg.shared_k_layout,
            )
        v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=value_cache_ptr,
                shape=(num_blocks * cfg.BLOCK_SIZE, cfg.NUM_KV_HEADS * cfg.HEAD_SIZE),
                strides=(stride_v_cache_1, stride_v_cache_3),
                block_shape=(cfg.BLOCK_SIZE, cfg.HEAD_SIZE),
                layout=kv_cfg.shared_v_layout,
            )
        
        return TDMKVLoader(kv_cfg,
         k_shared, v_shared, k_desc, v_desc, kv_head_idx, stride_k_cache_2, stride_v_cache_2)
    
    @gluon.jit
    def load_k_to_shared(self, k_offset, buffer_id):
        offsets=[
            (k_offset * (self.kv_cfg.BLOCK_SIZE)).to(gl.int32),
            (self.kv_head_idx * self.stride_k_cache_2).to(gl.int32),
        ]
        gl.amd.gfx1250.tdm.async_load(self.k_desc, offsets, self.k_shared.index(buffer_id))

    @gluon.jit
    def load_v_to_shared(self, v_offset, buffer_id):
        offsets=[
            (v_offset * (self.kv_cfg.BLOCK_SIZE)).to(gl.int32),
            (self.kv_head_idx * self.stride_v_cache_2).to(gl.int32),
        ]
        gl.amd.gfx1250.tdm.async_load(self.v_desc, offsets, self.v_shared.index(buffer_id))

    @gluon.jit
    def load_k_from_shared(self, wait_count, buffer_id):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        return self.k_shared.index(buffer_id).permute([1, 0]).load(layout=self.kv_cfg.k_reg_layout)

    @gluon.jit
    def load_v_from_shared(self, wait_count, buffer_id):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        return self.v_shared.index(buffer_id).load(layout=self.kv_cfg.v_reg_layout)

@aggregate
class AttentionConfig:
    """Configuration for unified attention layouts and derived constants (CDNA4)."""
    # Constants
    ARCH_NAME: gl.constexpr
    HEAD_SIZE: gl.constexpr
    BLOCK_SIZE: gl.constexpr
    BLOCK_M: gl.constexpr
    NUM_QUERY_HEADS: gl.constexpr
    NUM_KV_HEADS: gl.constexpr
    SLIDING_WINDOW: gl.constexpr
    NUM_QUERIES_PER_KV: gl.constexpr
    BLOCK_Q: gl.constexpr
    RCP_LN2: gl.constexpr
    QK_SCALE: gl.constexpr
    WARP_SIZE: gl.constexpr
    NUM_WARPS: gl.constexpr
    # Operator layouts
    qk_layout: gl.constexpr
    pv_layout: gl.constexpr

    # Dot operand layouts
    q_layout: gl.constexpr
    k_layout: gl.constexpr
    v_layout: gl.constexpr
    p_layout: gl.constexpr

    # Blocked layouts for global-to-shared loads
    blocked_q: gl.constexpr
    
    Q_CACHE_MODIFIER: gl.constexpr
    KV_CACHE_MODIFIER: gl.constexpr

    USE_LOAD_BUFFER_OP: gl.constexpr
    USE_STORE_BUFFER_OP: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, ARCH_NAME, NUM_WARPS,
                 HEAD_SIZE, BLOCK_SIZE, BLOCK_M, BLOCK_Q,
                 NUM_QUERY_HEADS, NUM_KV_HEADS, SLIDING_WINDOW,
                 SCALE, USE_LOAD_BUFFER_OP, USE_STORE_BUFFER_OP):

        # Constants
        self.HEAD_SIZE = gl.constexpr(HEAD_SIZE)
        self.BLOCK_SIZE = gl.constexpr(BLOCK_SIZE)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.NUM_QUERY_HEADS = gl.constexpr(NUM_QUERY_HEADS)
        self.NUM_KV_HEADS = gl.constexpr(NUM_KV_HEADS)
        self.SLIDING_WINDOW = gl.constexpr(SLIDING_WINDOW)
        # Derived constants
        self.NUM_QUERIES_PER_KV = gl.constexpr(NUM_QUERY_HEADS // NUM_KV_HEADS)
        self.BLOCK_Q = gl.constexpr(BLOCK_Q)
        self.RCP_LN2 = gl.constexpr(1.4426950408889634)
        self.QK_SCALE = gl.constexpr(SCALE * self.RCP_LN2)
        self.USE_LOAD_BUFFER_OP = gl.constexpr(USE_LOAD_BUFFER_OP)
        self.USE_STORE_BUFFER_OP = gl.constexpr(USE_STORE_BUFFER_OP)
        self.ARCH_NAME = gl.constexpr(ARCH_NAME)
        self.WARP_SIZE = gl.constexpr(32 if ARCH_NAME == "gfx1250" else 64)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        # Operator layouts (gfx1250 WMMA)
        if ARCH_NAME == "gfx1250":
            assert NUM_WARPS == 4 or NUM_WARPS == 8

            if NUM_WARPS == 4:
                warp_bases = [[1, 0], [2, 0]]
            else:
                warp_bases = [[1, 0], [2, 0], [4, 0]]
            self.qk_layout = gl.constexpr(
                gl.amd.AMDWMMALayout(version=3, transposed=True,
                                    instr_shape=[16, 16, 32], warp_bases=warp_bases))
        else:
            self.qk_layout = gl.constexpr(
                gl.amd.AMDMFMALayout(version=4, transposed=True,
                                    instr_shape=[32, 32, 16], warps_per_cta=[NUM_WARPS, 1]))
        self.pv_layout = self.qk_layout

        # Dot operand layouts
        self.q_layout = gl.constexpr(gl.DotOperandLayout(0, self.qk_layout, 8))
        self.k_layout = gl.constexpr(gl.DotOperandLayout(1, self.qk_layout, 8))
        self.v_layout = gl.constexpr(gl.DotOperandLayout(1, self.pv_layout, 8))
        self.p_layout = gl.constexpr(gl.DotOperandLayout(0, self.pv_layout, 8))

        # Blocked layouts for global-to-shared memory loads
        HEAD_SIZE_DIV = HEAD_SIZE // 8
        self.blocked_q = gl.constexpr(gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[self.WARP_SIZE // 8, 8],
            warps_per_cta=[NUM_WARPS, 1],
            order=[1, 0],
        ))

        self.Q_CACHE_MODIFIER = gl.constexpr(".cg")
        self.KV_CACHE_MODIFIER = gl.constexpr("")


@aggregate
class AttentionProgram:
    """Program state and core operations for the unified attention kernel."""

    cfg: AttentionConfig

    q: gl.tensor

    key_cache_ptr: gl.tensor
    value_cache_ptr: gl.tensor
    output_ptr: gl.tensor

    tile_start: gl.tensor
    tile_end: gl.tensor
    safe_tile_end: gl.tensor
    #query_pos_qk: gl.tensor
    query_mask_qk: gl.tensor
    #context_len: gl.tensor
    context_len_q_pos_qk: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg, q,
                 key_cache_ptr, value_cache_ptr, output_ptr,
                 tile_start, tile_end, safe_tile_end,
                 query_mask_qk, context_len_q_pos_qk):
        self.cfg = cfg
        self.q = q
        self.key_cache_ptr = key_cache_ptr
        self.value_cache_ptr = value_cache_ptr
        self.output_ptr = output_ptr
        self.tile_start = tile_start
        self.tile_end = tile_end
        self.safe_tile_end = safe_tile_end
        self.query_mask_qk = query_mask_qk
        self.context_len_q_pos_qk = context_len_q_pos_qk
    @gluon.jit    
    def initialize(cfg, q,
                 key_cache_ptr, value_cache_ptr, output_ptr,
                 max_seq_prefix_len,
                 q_block_local_idx, cur_batch_query_len, context_len,
                 query_pos, query_mask):
        # Calculate tile range
        num_tiles = (max_seq_prefix_len + cfg.BLOCK_SIZE - 1) // cfg.BLOCK_SIZE
        tile_start = 0
        tile_end = num_tiles
        if cfg.SLIDING_WINDOW > 0:
            qpos_lo = q_block_local_idx * cfg.BLOCK_Q
            qpos_hi = gl.minimum(
                qpos_lo + (cfg.BLOCK_M - 1) // cfg.NUM_QUERIES_PER_KV,
                cur_batch_query_len - 1,
            )
            first_allowed_key = context_len + qpos_lo - cfg.SLIDING_WINDOW + 1
            last_allowed_key = context_len + qpos_hi
            tile_start = gl.maximum(0, first_allowed_key // cfg.BLOCK_SIZE)
            tile_end = gl.minimum((last_allowed_key // cfg.BLOCK_SIZE) + 1, num_tiles)

        query_pos_qk = gl.convert_layout(query_pos, gl.SliceLayout(1, cfg.qk_layout))[:, None]
        query_mask_qk = gl.convert_layout(query_mask, cfg.qk_layout)

        context_len_q_pos_qk = context_len + query_pos_qk

        # Compute the tile index beyond which causal masking is needed.
        # min causal pos = context_len + first query pos in block
        # Tiles j < safe_tile_end have all KV positions within causal range
        # for every query row, so apply_mask_qk can be skipped.
        min_causal_pos = context_len + q_block_local_idx * cfg.BLOCK_Q
        safe_tile_end = (min_causal_pos + 1) // cfg.BLOCK_SIZE
        safe_tile_end = gl.minimum(safe_tile_end, tile_end)
        safe_tile_end = gl.maximum(safe_tile_end, tile_start)

        return AttentionProgram(cfg, q,
                 key_cache_ptr, value_cache_ptr, output_ptr,
                 tile_start, tile_end, safe_tile_end,
                 query_mask_qk, context_len_q_pos_qk)
    @gluon.jit
    def load_q_from_global(self, query_ptr, q_block_local_idx, 
    cur_batch_in_all_start_index, kv_head_idx, cur_batch_query_len, query_stride_0, query_stride_1):
        """Load Q from global memory."""
        offs_m = gl.arange(0, self.cfg.BLOCK_M, layout=gl.SliceLayout(1, self.cfg.q_layout))
        offs_d = gl.arange(0, self.cfg.HEAD_SIZE, layout=gl.SliceLayout(0, self.cfg.q_layout))
        query_pos = q_block_local_idx * self.cfg.BLOCK_Q + offs_m // self.cfg.NUM_QUERIES_PER_KV

        query_offset_0 = cur_batch_in_all_start_index + query_pos
        query_offset_1 = kv_head_idx * self.cfg.NUM_QUERIES_PER_KV + offs_m % self.cfg.NUM_QUERIES_PER_KV

        query_mask_0 = query_pos < cur_batch_query_len
        query_mask_1 = query_offset_1 < self.cfg.NUM_QUERY_HEADS
        query_mask = query_mask_0[:, None] & query_mask_1[:, None]

        q_offs = (query_offset_0[:, None] * query_stride_0 +
                query_offset_1[:, None] * query_stride_1 +
                offs_d[None, :])
        if self.cfg.USE_STORE_BUFFER_OP:
            q = gl.amd.cdna4.buffer_load(query_ptr + q_offs, mask=query_mask, other=0.0, cache_modifier=self.cfg.Q_CACHE_MODIFIER)
        else:
            q = gl.load(query_ptr + q_offs, mask=query_mask, other=0.0, cache_modifier=self.cfg.Q_CACHE_MODIFIER)
        return q, query_pos, query_mask


    @gluon.jit
    def compute_qk(self, k):
        S = gl.zeros([self.cfg.BLOCK_M, self.cfg.BLOCK_SIZE],
                     dtype=gl.float32, layout=self.cfg.qk_layout)
        if self.cfg.ARCH_NAME == "gfx1250":
            return gl.amd.gfx1250.wmma(self.q, k, S) * self.cfg.QK_SCALE
        else:
            return gl.amd.cdna4.mfma(self.q, k, S) * self.cfg.QK_SCALE
    
    @gluon.jit
    def apply_mask_qk(self, S, j):
        seq_offset = (j * self.cfg.BLOCK_SIZE
                      + gl.arange(0, self.cfg.BLOCK_SIZE, layout=gl.SliceLayout(0, self.cfg.qk_layout))[None, :])

        seq_mask = seq_offset <= self.context_len_q_pos_qk
        if self.cfg.SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((self.context_len_q_pos_qk - seq_offset) < self.cfg.SLIDING_WINDOW)
        full_mask = seq_mask
        S = gl.where(full_mask, S, float("-inf"))
        return S

    @gluon.jit
    def softmax_part0(self, S, M):
        m_ij = gl.maximum(M, gl.max(S, axis=1))
        m_ij = gl.where(m_ij > float("-inf"), m_ij, 0.0)
        p = gl.exp2(S - m_ij[:, None])
        alpha = gl.exp2(M - m_ij)
        return p, alpha, m_ij

    @gluon.jit
    def softmax_part1(self, p, L, acc, alpha):
        l_ij = gl.sum(p, 1)
        acc = acc * alpha[:, None]
        p = p.to(gl.bfloat16, fp_downcast_rounding="rtz")
        L = L * alpha + l_ij
        return p, L, acc

    @gluon.jit
    def compute_pv(self, p, v, acc):
        p = gl.convert_layout(p, self.cfg.p_layout)
        if self.cfg.ARCH_NAME == "gfx1250":
            return gl.amd.gfx1250.wmma(p, v, acc)
        else:
            return gl.amd.cdna4.mfma(p, v, acc)

    @gluon.jit
    def store_output(self, out, q_block_local_idx, cur_batch_in_all_start_index, kv_head_idx, 
    cur_batch_query_len, output_stride_0, output_stride_1):
        offs_m_out = gl.arange(0, self.cfg.BLOCK_M, layout=gl.SliceLayout(1, self.cfg.blocked_q))
        offs_d_out = gl.arange(0, self.cfg.HEAD_SIZE, layout=gl.SliceLayout(0, self.cfg.blocked_q))

        query_pos_out = q_block_local_idx * self.cfg.BLOCK_Q + offs_m_out // self.cfg.NUM_QUERIES_PER_KV
        query_offset_0_out = cur_batch_in_all_start_index + query_pos_out
        query_offset_1_out = kv_head_idx * self.cfg.NUM_QUERIES_PER_KV + offs_m_out % self.cfg.NUM_QUERIES_PER_KV

        o_offs = (query_offset_0_out[:, None] * output_stride_0 +
                query_offset_1_out[:, None] * output_stride_1 +
                offs_d_out[None, :])

        query_mask_0_out = query_pos_out < cur_batch_query_len
        query_mask_1_out = query_offset_1_out < self.cfg.NUM_QUERY_HEADS
        o_mask = query_mask_0_out[:, None] & query_mask_1_out[:, None]
        casted_out = out.to(self.output_ptr.dtype.element_ty)
        casted_out = gl.convert_layout(casted_out, self.cfg.blocked_q)
        if self.cfg.USE_STORE_BUFFER_OP:
            gl.amd.cdna4.buffer_store(casted_out, self.output_ptr, o_offs, mask=o_mask)
        else:
            gl.store(self.output_ptr + o_offs, casted_out, mask=o_mask)



@gluon.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: gl.constexpr,
):
    """Binary search to find the sequence index for a given query block index."""
    left = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = gl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@gluon.jit
def kernel_unified_attention_2d(
    query_ptr,            # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,        # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,      # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,             # [num_query_heads]
    output_ptr,           # [num_tokens, num_query_heads, head_size]
    block_tables_ptr,     # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,         # [num_seqs]
    query_start_len_ptr,  # [num_seqs+1]
    query_stride_0,
    query_stride_1,
    output_stride_0,
    output_stride_1,
    USE_SINKS: gl.constexpr,  # bool
    SLIDING_WINDOW: gl.constexpr,  # int
    num_blocks,
    stride_k_cache_0: gl.int32,
    stride_k_cache_1: gl.int32,
    stride_k_cache_2: gl.int32,
    stride_k_cache_3: gl.constexpr,
    stride_v_cache_0: gl.int32,
    stride_v_cache_1: gl.int32,
    stride_v_cache_2: gl.int32,
    stride_v_cache_3: gl.constexpr,
    block_table_stride,
    num_seqs: gl.constexpr,
    SCALE: gl.constexpr,
    NUM_QUERY_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_M: gl.constexpr,
    ARCH_NAME: gl.constexpr,
    USE_LOAD_BUFFER_OP: gl.constexpr = False,
    USE_STORE_BUFFER_OP: gl.constexpr = False,
    USE_TDM: gl.constexpr = False,
):
    NUM_WARPS: gl.constexpr = gl.num_warps()
    # Workgroup offsets
    kv_head_idx = gl.program_id(0)
    q_block_global_idx = gl.num_programs(1) - 1 - gl.program_id(1)
    # Build config with all layouts and derived constants
    cfg = AttentionConfig(ARCH_NAME, NUM_WARPS, 
                          HEAD_SIZE, BLOCK_SIZE, BLOCK_M, BLOCK_Q,
                          NUM_QUERY_HEADS, NUM_KV_HEADS, SLIDING_WINDOW, 
                          SCALE, USE_LOAD_BUFFER_OP, USE_STORE_BUFFER_OP)
    
    # Cast strides to int64 when not using buffer ops
    if not USE_LOAD_BUFFER_OP and not USE_TDM:
        stride_k_cache_0 = stride_k_cache_0.to(gl.int64)
        stride_k_cache_1 = stride_k_cache_1.to(gl.int64)
        stride_k_cache_2 = stride_k_cache_2.to(gl.int64)
        stride_v_cache_0 = stride_v_cache_0.to(gl.int64)
        stride_v_cache_1 = stride_v_cache_1.to(gl.int64)
        stride_v_cache_2 = stride_v_cache_2.to(gl.int64)

    if not USE_STORE_BUFFER_OP:
        output_stride_0 = output_stride_0.to(gl.int64)
        output_stride_1 = output_stride_1.to(gl.int64)
    

    # Find sequence index using binary search
    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, cfg.BLOCK_Q)

    # Get query block start and local index
    cur_batch_in_all_start_index = gl.load(query_start_len_ptr + seq_idx)
    q_block_start_idx = cur_batch_in_all_start_index // cfg.BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_stop_index = gl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * cfg.BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout))
    offs_d = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, cfg.q_layout))
    query_pos = q_block_local_idx * cfg.BLOCK_Q + offs_m // cfg.NUM_QUERIES_PER_KV

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_m % cfg.NUM_QUERIES_PER_KV

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < NUM_QUERY_HEADS
    query_mask = query_mask_0[:, None] & query_mask_1[:, None]

    q_offs = (query_offset_0[:, None] * query_stride_0 +
              query_offset_1[:, None] * query_stride_1 +
              offs_d[None, :])

    q = gl.load(query_ptr + q_offs, mask=query_mask, other=0.0, cache_modifier=cfg.Q_CACHE_MODIFIER)

    seq_len = gl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len
    block_tables_ptr_shifted = block_tables_ptr + seq_idx * block_table_stride

    # Max KV position that any query in this block attends to
    max_seq_prefix_len = (context_len
                          + q_block_local_idx * cfg.BLOCK_Q
                          + (BLOCK_M - 1) // cfg.NUM_QUERIES_PER_KV + 1)
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)

    # build program
    pgm = AttentionProgram.initialize(cfg, q,
                                      key_cache_ptr, value_cache_ptr, output_ptr,
                                      max_seq_prefix_len,
                                      q_block_local_idx, cur_batch_query_len, context_len,
                                      query_pos, query_mask,
                                      )
    if USE_TDM:
        KVLoader: gl.constexpr = TDMKVLoader
    else:
        KVLoader: gl.constexpr = AsyncKVLoader

    kv_loader = KVLoader.initialize(cfg, key_cache_ptr, value_cache_ptr, kv_head_idx,
                                    num_blocks,
                                    stride_k_cache_1, stride_k_cache_2, stride_k_cache_3,
                                    stride_v_cache_1, stride_v_cache_2, stride_v_cache_3)
        
    # Initialize accumulators
    if not USE_SINKS:
        M = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32,
                    layout=gl.SliceLayout(1, cfg.pv_layout))
    else:
        offs_m_pv = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout))
        query_offset_1_pv = (kv_head_idx * cfg.NUM_QUERIES_PER_KV
                             + offs_m_pv % cfg.NUM_QUERIES_PER_KV)
        query_mask_1_pv = query_offset_1_pv < NUM_QUERY_HEADS
        # Prescale with RCP_LN2, needed for exp2
        M = (
            gl.load(
                sink_ptr + query_offset_1_pv,
                mask=query_mask_1_pv,
                other=float("-inf"),
            ).to(dtype=gl.float32)
            * cfg.RCP_LN2
        )

    L = gl.full([BLOCK_M], 1.0, dtype=gl.float32,
                layout=gl.SliceLayout(1, cfg.pv_layout))
    acc = gl.zeros([BLOCK_M, HEAD_SIZE], dtype=gl.float32, layout=cfg.pv_layout)
    # TODO (cagri): Assuming stride_k_cache_0 == stride_v_cache_0
    # Prologue: load first tile's block index and issue async K, V loads
    physical_block_idx = gl.load(block_tables_ptr_shifted + pgm.tile_start)
    if not USE_TDM:
        physical_block_idx = physical_block_idx * stride_k_cache_0
    # rotating buffer index logic
    # TODO (cagri): Loop unrolling can get rid of this
    buffer_id: gl.int32 = 0
    kv_loader.load_k_to_shared(physical_block_idx, buffer_id=buffer_id)
    kv_loader.load_v_to_shared(physical_block_idx, buffer_id=buffer_id)

    # Main attention loop over KV tiles (staged, num_stages=2)
    for j in range(pgm.tile_start, pgm.tile_end - 1):
        next_physical_block_idx = gl.load(block_tables_ptr_shifted + j + 1)
        if not USE_TDM:
            next_physical_block_idx = next_physical_block_idx * stride_k_cache_0
        k = kv_loader.load_k_from_shared(wait_count=1, buffer_id=buffer_id)

        # Prefetch next tile (shared is free since k, v are in registers)
        kv_loader.load_k_to_shared(next_physical_block_idx, buffer_id=1-buffer_id)
        kv_loader.load_v_to_shared(next_physical_block_idx, buffer_id=1-buffer_id)

        # Compute attention for current tile
        S = pgm.compute_qk(k)
        if j >= pgm.safe_tile_end or SLIDING_WINDOW > 0:
            S = pgm.apply_mask_qk(S, j)
        p, alpha, M = pgm.softmax_part0(S, M)
        p, L, acc = pgm.softmax_part1(p, L, acc, alpha)
        v = kv_loader.load_v_from_shared(wait_count=2, buffer_id=buffer_id)
        acc = pgm.compute_pv(p, v, acc)
        buffer_id = 1 - buffer_id
        
    # Load k_i, v_i from shared into registers
    k = kv_loader.load_k_from_shared(wait_count=1, buffer_id=buffer_id)
    # Compute attention for current tile
    S = pgm.compute_qk(k)
    S = pgm.apply_mask_qk(S, pgm.tile_end - 1)
    p, alpha, M = pgm.softmax_part0(S, M)
    p, L, acc = pgm.softmax_part1(p, L, acc, alpha)
    v = kv_loader.load_v_from_shared(wait_count=0, buffer_id=buffer_id)
    acc = pgm.compute_pv(p, v, acc)
    # Normalize and store output
    l_recip = 1 / L[:, None]
    acc = acc * l_recip


    pgm.store_output(acc, q_block_local_idx, cur_batch_in_all_start_index, kv_head_idx, 
    cur_batch_query_len, output_stride_0, output_stride_1)


def unified_attention(q,
                    k,
                    v,
                    out,
                    cu_seqlens_q,
                    seqused_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale,
                    causal,
                    window_size,
                    block_table,
                    softcap,
                    q_descale,
                    k_descale,
                    v_descale,
                    sinks):
    """
    Run the unified attention kernel with paged KV cache.

    Args:
        q: Query tensor [num_tokens, num_query_heads, head_size]
        k: Key cache [num_blks, blk_size, num_kv_heads, head_size]
        v: Value cache [num_blks, blk_size, num_kv_heads, head_size]
        out: Output tensor [num_tokens, num_query_heads, head_size]
        cu_seqlens_q: Cumulative query lengths [num_seqs + 1]
        seqused_k: Sequence lengths [num_seqs]
        max_seqlen_q: Maximum query length
        max_seqlen_k: Maximum key/value length
        softmax_scale: Attention scale factor
        causal: Whether to use causal masking
        window_size: Sliding window size
        block_table: Block tables [num_seqs, max_num_blocks_per_seq]
        softcap: Softcap value
        q_descale: Query scale
        k_descale: Key scale
        v_descale: Value scale
        sinks: Sinks tensor [num_query_heads,]
    """
    NUM_SEQS = len(seqused_k)
    NUM_Q_HEADS = q.shape[1]
    NUM_KV_HEADS = k.shape[2]
    HEAD_SIZE = q.shape[2]
    BLOCK_SIZE = k.shape[1]
    BLOCK_M = 128
    SLIDING_WINDOW = 1 + window_size[0]
    num_blocks = k.shape[0]

    NUM_QUERIES_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
    BLOCK_Q = BLOCK_M // NUM_QUERIES_PER_KV
    total_query_blocks = q.shape[0] // BLOCK_Q + NUM_SEQS

    ARCH_NAME = arch_info.get_arch()
    NUM_WARPS = 4
    USE_TDM = ARCH_NAME == "gfx1250"
    kv_size = k.nelement() * k.element_size()
    MAX_INT32 = 2**31 - 1
    USE_LOAD_BUFFER_OP = kv_size <= MAX_INT32
    USE_STORE_BUFFER_OP = out.nelement() * out.element_size() <= MAX_INT32
    waves_per_eu = 4 if HEAD_SIZE < 128 else 2
    grid = (NUM_KV_HEADS, total_query_blocks)
    attn_kernel = kernel_unified_attention_2d[grid](
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        output_ptr=out,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        query_start_len_ptr=cu_seqlens_q,
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=SLIDING_WINDOW,
        num_blocks=num_blocks,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        block_table_stride=block_table.stride(0),
        num_seqs=NUM_SEQS,
        SCALE=softmax_scale,
        NUM_QUERY_HEADS=NUM_Q_HEADS,
        NUM_KV_HEADS=NUM_KV_HEADS,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        ARCH_NAME=ARCH_NAME,
        waves_per_eu=waves_per_eu,
        USE_LOAD_BUFFER_OP=USE_LOAD_BUFFER_OP,
        USE_STORE_BUFFER_OP=USE_STORE_BUFFER_OP,
        num_warps=NUM_WARPS,
        USE_TDM=USE_TDM,
    )
    if PRINT_IRS and getattr(unified_attention, "print", False) == False:
        setattr(unified_attention, "print", True)
        print_irs_to_files(attn_kernel, "unified_attention_2d_gluon")
    return attn_kernel


def print_irs_to_files(compiled_kernel, prefix):
    for key in compiled_kernel.asm.keys():
        with open(f"{prefix}_{key}.txt", "w") as fptr:
            print(compiled_kernel.asm[key], file=fptr)
