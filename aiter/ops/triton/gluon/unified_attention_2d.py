import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.language.core import _aggregate as aggregate
import pytest
from aiter.ops.triton.utils._triton import arch_info
import os
from aiter.ops.triton.utils.types import e4m3_dtype

float8_info = torch.finfo(e4m3_dtype)
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
    K_WIDTH: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, cfg):
        # Blocked layouts for global-to-shared memory loads
        HEAD_SIZE_DIV = cfg.HEAD_SIZE // 8
        # gl.static_assert(WARP_SIZE % HEAD_SIZE_DIV == 0, "WARP_SIZE must be divisible by HEAD_SIZE_DIV")
        self.blocked_v = gl.constexpr(
            gl.BlockedLayout(
                size_per_thread=[1, 8],
                threads_per_warp=[cfg.WARP_SIZE // HEAD_SIZE_DIV, HEAD_SIZE_DIV],
                warps_per_cta=[cfg.NUM_WARPS, 1],
                order=[1, 0],
            )
        )
        self.blocked_k = gl.constexpr(
            gl.BlockedLayout(
                size_per_thread=[8, 1],
                threads_per_warp=[HEAD_SIZE_DIV, cfg.WARP_SIZE // HEAD_SIZE_DIV],
                warps_per_cta=[1, cfg.NUM_WARPS],
                order=[0, 1],
            )
        )
        if cfg.SHUFFLED_KV_CACHE:
            self.shared_k_layout = gl.constexpr(
                gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
            )
            self.shared_v_layout = gl.constexpr(
                gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
            )
        else:
            self.shared_k_layout = gl.constexpr(
                gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[0, 1])
            )
            self.shared_v_layout = gl.constexpr(
                gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
            )

        self.KV_CACHE_MODIFIER = cfg.KV_CACHE_MODIFIER
        self.USE_LOAD_BUFFER_OP = cfg.USE_LOAD_BUFFER_OP

        self.k_reg_layout = gl.constexpr(cfg.k_layout)
        self.v_reg_layout = gl.constexpr(cfg.v_layout)
        self.K_WIDTH = gl.constexpr(cfg.K_WIDTH)


@aggregate
class AsyncKVLoader:
    kv_cfg: AsyncKVLoaderConfig
    key_cache_ptr: gl.tensor
    value_cache_ptr: gl.tensor
    block_tables_ptr_shifted: gl.tensor
    k_shared: gl.shared_memory_descriptor
    v_shared: gl.shared_memory_descriptor
    k_base_offset: gl.tensor
    v_base_offset: gl.tensor
    stride_k_cache_0: gl.tensor
    stride_v_cache_0: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        kv_cfg,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr_shifted,
        k_shared,
        v_shared,
        k_base_offset,
        v_base_offset,
        stride_k_cache_0,
        stride_v_cache_0,
    ):
        self.kv_cfg = kv_cfg
        self.key_cache_ptr = key_cache_ptr
        self.value_cache_ptr = value_cache_ptr
        self.k_shared = k_shared
        self.v_shared = v_shared
        self.k_base_offset = k_base_offset
        self.v_base_offset = v_base_offset
        self.block_tables_ptr_shifted = block_tables_ptr_shifted
        self.stride_k_cache_0 = stride_k_cache_0
        self.stride_v_cache_0 = stride_v_cache_0

    @gluon.jit
    def initialize(
        cfg,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr_shifted,
        kv_head_idx,
        num_blocks,
        stride_k_cache_0,
        stride_k_cache_1,
        stride_k_cache_2,
        stride_k_cache_3,
        stride_v_cache_0,
        stride_v_cache_1,
        stride_v_cache_2,
        stride_v_cache_3,
    ):
        kv_cfg = AsyncKVLoaderConfig(cfg)
        k_shared = gl.allocate_shared_memory(
            key_cache_ptr.type.element_ty,
            [2, cfg.HEAD_SIZE, cfg.TILE_SIZE],
            layout=kv_cfg.shared_k_layout,
        )
        v_shared = gl.allocate_shared_memory(
            value_cache_ptr.type.element_ty,
            [2, cfg.TILE_SIZE, cfg.HEAD_SIZE],
            layout=kv_cfg.shared_v_layout,
        )

        # Precompute KV load offsets (constant across tiles)
        offs_d_k = gl.arange(
            0, cfg.HEAD_SIZE, layout=gl.SliceLayout(1, kv_cfg.blocked_k)
        )[:, None]
        offs_n_k = gl.arange(
            0, cfg.TILE_SIZE, layout=gl.SliceLayout(0, kv_cfg.blocked_k)
        )[None, :]
        k_base_offset = (
            kv_head_idx * stride_k_cache_2
            + offs_d_k * stride_k_cache_3
            + offs_n_k * stride_k_cache_1
        )

        offs_d_v = gl.arange(
            0, cfg.HEAD_SIZE, layout=gl.SliceLayout(0, kv_cfg.blocked_v)
        )[None, :]
        offs_n_v = gl.arange(
            0, cfg.TILE_SIZE, layout=gl.SliceLayout(1, kv_cfg.blocked_v)
        )[:, None]
        v_base_offset = (
            kv_head_idx * stride_v_cache_2
            + offs_d_v * stride_v_cache_3
            + offs_n_v * stride_v_cache_1
        )

        return AsyncKVLoader(
            kv_cfg,
            key_cache_ptr,
            value_cache_ptr,
            block_tables_ptr_shifted,
            k_shared,
            v_shared,
            k_base_offset,
            v_base_offset,
            stride_k_cache_0,
            stride_v_cache_0,
        )

    @gluon.jit
    def load_k_to_shared(self, k_offset, buffer_id):
        # Async copy K tile from global to shared memory
        if self.kv_cfg.USE_LOAD_BUFFER_OP:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                self.k_shared.index(buffer_id),
                self.key_cache_ptr,
                self.k_base_offset + k_offset,
                cache_modifier=self.kv_cfg.KV_CACHE_MODIFIER,
            )
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(
                self.k_shared.index(buffer_id),
                self.key_cache_ptr + self.k_base_offset + k_offset,
                cache_modifier=self.kv_cfg.KV_CACHE_MODIFIER,
            )
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def load_v_to_shared(self, v_offset, buffer_id):
        # Async copy V tile from global to shared memory
        if self.kv_cfg.USE_LOAD_BUFFER_OP:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                self.v_shared.index(buffer_id),
                self.value_cache_ptr,
                self.v_base_offset + v_offset,
                cache_modifier=self.kv_cfg.KV_CACHE_MODIFIER,
            )
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(
                self.v_shared.index(buffer_id),
                self.value_cache_ptr + self.v_base_offset + v_offset,
                cache_modifier=self.kv_cfg.KV_CACHE_MODIFIER,
            )
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def load_k_from_shared(self, wait_count, target_dtype, buffer_id):
        # Wait for async K copy and load from shared memory
        gl.amd.cdna4.async_copy.wait_group(wait_count)
        return gl.amd.cdna4.async_copy.load_shared_relaxed(
            self.k_shared.index(buffer_id), self.kv_cfg.k_reg_layout
        ).to(target_dtype)

    @gluon.jit
    def load_v_from_shared(self, wait_count, target_dtype, buffer_id):
        # Wait for async V copy and load from shared memory
        gl.amd.cdna4.async_copy.wait_group(wait_count)
        return gl.amd.cdna4.async_copy.load_shared_relaxed(
            self.v_shared.index(buffer_id), self.kv_cfg.v_reg_layout
        ).to(target_dtype)

    @gluon.jit
    def load_block_ids(self, i):
        return gl.load(self.block_tables_ptr_shifted + i) * self.stride_k_cache_0


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
    HEAD_SIZE: gl.constexpr
    NUM_KV_HEADS: gl.constexpr
    K_WIDTH: gl.constexpr
    SHUFFLED_KV_CACHE: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, cfg):
        # Swizzled shared memory layouts for K and V
        if cfg.SHUFFLED_KV_CACHE:
            self.shared_k_layout = gl.constexpr(
                gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
            )
            self.shared_v_layout = gl.constexpr(
                gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
            )
        else:
            padding = 8 if cfg.Q_FP8 else 8
            self.shared_k_layout = gl.constexpr(
                gl.PaddedSharedLayout.with_identity_for(
                    [[cfg.HEAD_SIZE, padding]], [cfg.BLOCK_SIZE, cfg.HEAD_SIZE], [1, 0]
                )
            )
            self.shared_v_layout = gl.constexpr(
                gl.PaddedSharedLayout.with_identity_for(
                    [[cfg.HEAD_SIZE, 16]], [cfg.BLOCK_SIZE, cfg.HEAD_SIZE], [1, 0]
                )
            )
        self.KV_CACHE_MODIFIER = gl.constexpr(cfg.KV_CACHE_MODIFIER)
        self.USE_LOAD_BUFFER_OP = gl.constexpr(cfg.USE_LOAD_BUFFER_OP)
        self.SHUFFLED_KV_CACHE = gl.constexpr(cfg.SHUFFLED_KV_CACHE)
        self.k_reg_layout = gl.constexpr(cfg.k_layout)
        self.v_reg_layout = gl.constexpr(cfg.v_layout)
        self.BLOCK_SIZE = gl.constexpr(cfg.BLOCK_SIZE)
        self.HEAD_SIZE = gl.constexpr(cfg.HEAD_SIZE)
        self.K_WIDTH = gl.constexpr(cfg.K_WIDTH)
        self.NUM_KV_HEADS = gl.constexpr(cfg.NUM_KV_HEADS)


@aggregate
class TDMKVLoader:
    kv_cfg: TDMKVLoaderConfig
    block_tables_ptr_shifted: gl.tensor
    k_shared: gl.shared_memory_descriptor
    v_shared: gl.shared_memory_descriptor
    k_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    v_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    kv_head_idx: gl.tensor
    stride_k_cache_2: gl.tensor
    stride_v_cache_2: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        kv_cfg,
        block_tables_ptr_shifted,
        k_shared,
        v_shared,
        k_desc,
        v_desc,
        kv_head_idx,
        stride_k_cache_2,
        stride_v_cache_2,
    ):
        self.kv_cfg = kv_cfg
        self.k_shared = k_shared
        self.v_shared = v_shared
        self.k_desc = k_desc
        self.v_desc = v_desc
        self.block_tables_ptr_shifted = block_tables_ptr_shifted
        self.kv_head_idx = kv_head_idx
        self.stride_k_cache_2 = stride_k_cache_2
        self.stride_v_cache_2 = stride_v_cache_2

    @gluon.jit
    def initialize(
        cfg,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr_shifted,
        kv_head_idx,
        num_blocks,
        stride_k_cache_0,
        stride_k_cache_1,
        stride_k_cache_2,
        stride_k_cache_3,
        stride_v_cache_0,
        stride_v_cache_1,
        stride_v_cache_2,
        stride_v_cache_3,
    ):
        kv_cfg = TDMKVLoaderConfig(cfg)
        # if cfg.SHUFFLED_KV_CACHE:
        #     block_shape = [1, cfg.BLOCK_SIZE * cfg.HEAD_SIZE]
        # else:
        k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=key_cache_ptr,
            shape=(
                (num_blocks * cfg.NUM_KV_HEADS, cfg.BLOCK_SIZE * cfg.HEAD_SIZE)
                if cfg.SHUFFLED_KV_CACHE
                else (num_blocks * cfg.BLOCK_SIZE, cfg.NUM_KV_HEADS * cfg.HEAD_SIZE)
            ),
            strides=(stride_k_cache_1, stride_k_cache_3),
            block_shape=(
                (1, cfg.BLOCK_SIZE * cfg.HEAD_SIZE)
                if cfg.SHUFFLED_KV_CACHE
                else (cfg.BLOCK_SIZE, cfg.HEAD_SIZE)
            ),
            layout=kv_cfg.shared_k_layout,
        )
        v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=value_cache_ptr,
            shape=(
                (num_blocks * cfg.NUM_KV_HEADS, cfg.BLOCK_SIZE * cfg.HEAD_SIZE)
                if cfg.SHUFFLED_KV_CACHE
                else (num_blocks * cfg.BLOCK_SIZE, cfg.NUM_KV_HEADS * cfg.HEAD_SIZE)
            ),
            strides=(stride_v_cache_1, stride_v_cache_3),
            block_shape=(
                (1, cfg.BLOCK_SIZE * cfg.HEAD_SIZE)
                if cfg.SHUFFLED_KV_CACHE
                else (cfg.BLOCK_SIZE, cfg.HEAD_SIZE)
            ),
            layout=kv_cfg.shared_v_layout,
        )

        k_shared = gl.allocate_shared_memory(
            key_cache_ptr.type.element_ty,
            [2] + k_desc.block_shape,
            layout=kv_cfg.shared_k_layout,
        )
        v_shared = gl.allocate_shared_memory(
            value_cache_ptr.type.element_ty,
            [2] + v_desc.block_shape,
            layout=kv_cfg.shared_v_layout,
        )

        return TDMKVLoader(
            kv_cfg,
            block_tables_ptr_shifted,
            k_shared,
            v_shared,
            k_desc,
            v_desc,
            kv_head_idx,
            stride_k_cache_2,
            stride_v_cache_2,
        )

    @gluon.jit
    def load_k_to_shared(self, k_offset, buffer_id):
        if self.kv_cfg.SHUFFLED_KV_CACHE:
            offsets = [
                (k_offset * self.kv_cfg.NUM_KV_HEADS + self.kv_head_idx).to(gl.int32),
                0,
            ]
        else:
            offsets = [
                (k_offset * (self.kv_cfg.BLOCK_SIZE)).to(gl.int32),
                (self.kv_head_idx * self.stride_k_cache_2).to(gl.int32),
            ]
        gl.amd.gfx1250.tdm.async_load(
            self.k_desc, offsets, self.k_shared.index(buffer_id)
        )

    @gluon.jit
    def load_v_to_shared(self, v_offset, buffer_id):
        if self.kv_cfg.SHUFFLED_KV_CACHE:
            offsets = [
                (v_offset * self.kv_cfg.NUM_KV_HEADS + self.kv_head_idx).to(gl.int32),
                0,
            ]
        else:
            offsets = [
                (v_offset * (self.kv_cfg.BLOCK_SIZE)).to(gl.int32),
                (self.kv_head_idx * self.stride_v_cache_2).to(gl.int32),
            ]
        gl.amd.gfx1250.tdm.async_load(
            self.v_desc, offsets, self.v_shared.index(buffer_id)
        )

    @gluon.jit
    def load_k_from_shared(self, wait_count, target_dtype, buffer_id):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        if self.kv_cfg.SHUFFLED_KV_CACHE:
            return (
                self.lds_unshuffle_k(buffer_id).load(layout=self.kv_cfg.k_reg_layout)
            ).to(target_dtype)
        else:
            return (
                self.k_shared.index(buffer_id)
                .permute([1, 0])
                .load(layout=self.kv_cfg.k_reg_layout)
            ).to(target_dtype)

    @gluon.jit
    def load_v_from_shared(self, wait_count, target_dtype, buffer_id):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        if self.kv_cfg.SHUFFLED_KV_CACHE:
            return (
                self.lds_unshuffle_v(buffer_id).load(layout=self.kv_cfg.v_reg_layout)
            ).to(target_dtype)
        else:
            return (
                self.v_shared.index(buffer_id)
                .load(layout=self.kv_cfg.v_reg_layout)
                .to(target_dtype)
            )

    @gluon.jit
    def load_block_ids(self, i):
        return gl.load(self.block_tables_ptr_shifted + i)

    @gluon.jit
    def lds_unshuffle_k(self, buffer_id):
        return (
            self.k_shared.index(buffer_id)
            .reshape(
                (
                    1,
                    self.kv_cfg.BLOCK_SIZE // 16,
                    self.kv_cfg.HEAD_SIZE // (2 * self.kv_cfg.K_WIDTH),
                    2,
                    16,
                    self.kv_cfg.K_WIDTH,
                )
            )
            .permute((0, 1, 4, 2, 3, 5))
            .reshape((self.kv_cfg.BLOCK_SIZE, self.kv_cfg.HEAD_SIZE))
            .permute((1, 0))
        )

    @gluon.jit
    def lds_unshuffle_v(self, buffer_id):
        return (
            self.v_shared.index(buffer_id)
            .reshape(
                (
                    1,
                    self.kv_cfg.HEAD_SIZE // 16,
                    self.kv_cfg.BLOCK_SIZE // (2 * self.kv_cfg.K_WIDTH),
                    2,
                    16,
                    self.kv_cfg.K_WIDTH,
                )
            )
            .permute((0, 1, 4, 2, 3, 5))
            .reshape(
                (
                    1,
                    self.kv_cfg.HEAD_SIZE,
                    self.kv_cfg.BLOCK_SIZE,
                )
            )
            .permute((1, 0, 2))
            .reshape((self.kv_cfg.HEAD_SIZE, self.kv_cfg.BLOCK_SIZE))
            .permute((1, 0))
        )


@aggregate
class TDMGatherKVLoaderConfig:
    """Configuration for TDM KV loader."""

    shared_k_layout: gl.constexpr
    shared_v_layout: gl.constexpr
    USE_LOAD_BUFFER_OP: gl.constexpr
    KV_CACHE_MODIFIER: gl.constexpr

    k_reg_layout: gl.constexpr
    v_reg_layout: gl.constexpr
    BLOCK_SIZE: gl.constexpr
    HEAD_SIZE: gl.constexpr
    NUM_KV_HEADS: gl.constexpr
    NUM_KV_BLOCKS: gl.constexpr
    TILE_SIZE: gl.constexpr
    gather_ids_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, cfg):
        # Swizzled shared memory layouts for K and V
        self.shared_k_layout = gl.constexpr(
            gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
        )
        self.shared_v_layout = gl.constexpr(
            gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
        )

        self.KV_CACHE_MODIFIER = cfg.KV_CACHE_MODIFIER
        self.USE_LOAD_BUFFER_OP = cfg.USE_LOAD_BUFFER_OP

        self.k_reg_layout = gl.constexpr(cfg.k_layout)
        self.v_reg_layout = gl.constexpr(cfg.v_layout)
        self.BLOCK_SIZE = gl.constexpr(cfg.BLOCK_SIZE)
        self.HEAD_SIZE = gl.constexpr(cfg.HEAD_SIZE)
        self.NUM_KV_BLOCKS = gl.constexpr(cfg.NUM_KV_BLOCKS)
        self.TILE_SIZE = gl.constexpr(cfg.TILE_SIZE)
        self.NUM_KV_HEADS = gl.constexpr(cfg.NUM_KV_HEADS)

        self.gather_ids_layout = gl.constexpr(
            gl.BlockedLayout(
                size_per_thread=[cfg.NUM_KV_BLOCKS],
                threads_per_warp=[cfg.WARP_SIZE],
                warps_per_cta=[cfg.NUM_WARPS],
                order=[0],
            )
        )


@aggregate
class TDMGatherKVLoader:
    kv_cfg: TDMGatherKVLoaderConfig
    block_tables_ptr_shifted: gl.tensor
    k_shared: gl.shared_memory_descriptor
    v_shared: gl.shared_memory_descriptor
    k_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    v_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    kv_head_idx: gl.tensor
    stride_k_cache_2: gl.tensor
    stride_v_cache_2: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        kv_cfg,
        block_tables_ptr_shifted,
        k_shared,
        v_shared,
        k_desc,
        v_desc,
        kv_head_idx,
        stride_k_cache_2,
        stride_v_cache_2,
    ):
        self.kv_cfg = kv_cfg
        self.k_shared = k_shared
        self.v_shared = v_shared
        self.k_desc = k_desc
        self.v_desc = v_desc
        self.block_tables_ptr_shifted = block_tables_ptr_shifted
        self.kv_head_idx = kv_head_idx
        self.stride_k_cache_2 = stride_k_cache_2
        self.stride_v_cache_2 = stride_v_cache_2

    @gluon.jit
    def initialize(
        cfg,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr_shifted,
        kv_head_idx,
        num_blocks,
        stride_k_cache_0,
        stride_k_cache_1,
        stride_k_cache_2,
        stride_k_cache_3,
        stride_v_cache_0,
        stride_v_cache_1,
        stride_v_cache_2,
        stride_v_cache_3,
    ):
        kv_cfg = TDMGatherKVLoaderConfig(cfg)
        k_shared = gl.allocate_shared_memory(
            key_cache_ptr.type.element_ty,
            [2, cfg.NUM_KV_BLOCKS, cfg.BLOCK_SIZE * cfg.HEAD_SIZE],
            layout=kv_cfg.shared_k_layout,
        )
        v_shared = gl.allocate_shared_memory(
            value_cache_ptr.type.element_ty,
            [2, cfg.NUM_KV_BLOCKS, cfg.BLOCK_SIZE * cfg.HEAD_SIZE],
            layout=kv_cfg.shared_v_layout,
        )

        k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=key_cache_ptr,
            shape=(num_blocks * cfg.NUM_KV_HEADS, cfg.BLOCK_SIZE * cfg.HEAD_SIZE),
            strides=(stride_k_cache_1, stride_k_cache_3),
            block_shape=(cfg.NUM_KV_BLOCKS, cfg.BLOCK_SIZE * cfg.HEAD_SIZE),
            layout=kv_cfg.shared_k_layout,
        )
        v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=value_cache_ptr,
            shape=(num_blocks * cfg.NUM_KV_HEADS, cfg.BLOCK_SIZE * cfg.HEAD_SIZE),
            strides=(stride_v_cache_1, stride_v_cache_3),
            block_shape=(cfg.NUM_KV_BLOCKS, cfg.BLOCK_SIZE * cfg.HEAD_SIZE),
            layout=kv_cfg.shared_v_layout,
        )

        return TDMGatherKVLoader(
            kv_cfg,
            block_tables_ptr_shifted,
            k_shared,
            v_shared,
            k_desc,
            v_desc,
            kv_head_idx,
            stride_k_cache_2,
            stride_v_cache_2,
        )

    @gluon.jit
    def load_k_to_shared(self, k_offset, buffer_id):
        src_row_indices = (k_offset * self.kv_cfg.NUM_KV_HEADS + self.kv_head_idx).to(
            gl.int32
        )

        gl.amd.gfx1250.tdm.async_gather(
            self.k_desc, src_row_indices, 0, self.k_shared.index(buffer_id)
        )

    @gluon.jit
    def load_v_to_shared(self, v_offset, buffer_id):
        src_row_indices = (v_offset * self.kv_cfg.NUM_KV_HEADS + self.kv_head_idx).to(
            gl.int32
        )
        gl.amd.gfx1250.tdm.async_gather(
            self.v_desc, src_row_indices, 0, self.v_shared.index(buffer_id)
        )

    @gluon.jit
    def load_k_from_shared(self, wait_count, target_dtype, buffer_id):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        return (
            self.k_shared.index(buffer_id)
            .reshape([self.kv_cfg.TILE_SIZE, self.kv_cfg.HEAD_SIZE])
            .permute([1, 0])
            .load(layout=self.kv_cfg.k_reg_layout)
        ).to(target_dtype)

    @gluon.jit
    def load_v_from_shared(self, wait_count, target_dtype, buffer_id):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        return (
            self.v_shared.index(buffer_id)
            .reshape([self.kv_cfg.TILE_SIZE, self.kv_cfg.HEAD_SIZE])
            .load(layout=self.kv_cfg.v_reg_layout)
        ).to(target_dtype)

    @gluon.jit
    def load_block_ids(self, i):
        offs = gl.arange(
            0, self.kv_cfg.NUM_KV_BLOCKS, layout=self.kv_cfg.gather_ids_layout
        )
        return gl.load(
            self.block_tables_ptr_shifted + i * self.kv_cfg.NUM_KV_BLOCKS + offs
        )


@aggregate
class AttentionConfig:
    """Configuration for unified attention layouts and derived constants (CDNA4)."""

    # Constants
    ARCH_NAME: gl.constexpr
    HEAD_SIZE: gl.constexpr
    BLOCK_SIZE: gl.constexpr
    BLOCK_M: gl.constexpr
    TILE_SIZE: gl.constexpr
    NUM_KV_BLOCKS: gl.constexpr
    NUM_QUERY_HEADS: gl.constexpr
    NUM_KV_HEADS: gl.constexpr
    SLIDING_WINDOW: gl.constexpr
    NUM_QUERIES_PER_KV: gl.constexpr
    BLOCK_Q: gl.constexpr
    RCP_LN2: gl.constexpr
    QK_SCALE: gl.constexpr
    SOFTMAX_SCALE: gl.constexpr
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
    ALL_DECODE: gl.constexpr
    SHUFFLED_KV_CACHE: gl.constexpr

    Q_FP8: gl.constexpr
    KV_FP8: gl.constexpr
    K_WIDTH: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        ARCH_NAME,
        NUM_WARPS,
        HEAD_SIZE,
        BLOCK_SIZE,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        SLIDING_WINDOW,
        SCALE,
        USE_LOAD_BUFFER_OP,
        USE_STORE_BUFFER_OP,
        ALL_DECODE,
        SHUFFLED_KV_CACHE,
        Q_FP8,
        KV_FP8,
    ):

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
        self.NUM_KV_BLOCKS = gl.constexpr(TILE_SIZE // BLOCK_SIZE)
        self.TILE_SIZE = gl.constexpr(TILE_SIZE)
        self.RCP_LN2 = gl.constexpr(1.4426950408889634)
        self.QK_SCALE = gl.constexpr(self.RCP_LN2 * SCALE)
        self.SOFTMAX_SCALE = gl.constexpr(SCALE)
        self.USE_LOAD_BUFFER_OP = gl.constexpr(USE_LOAD_BUFFER_OP)
        self.USE_STORE_BUFFER_OP = gl.constexpr(USE_STORE_BUFFER_OP)
        self.ALL_DECODE = gl.constexpr(ALL_DECODE)
        self.SHUFFLED_KV_CACHE = gl.constexpr(SHUFFLED_KV_CACHE)
        self.Q_FP8 = gl.constexpr(Q_FP8)
        self.KV_FP8 = gl.constexpr(KV_FP8)
        self.ARCH_NAME = gl.constexpr(ARCH_NAME)
        self.WARP_SIZE = gl.constexpr(32 if ARCH_NAME == "gfx1250" else 64)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        FP8_DOT = gl.constexpr(Q_FP8 and KV_FP8)
        self.K_WIDTH = gl.constexpr(8)
        # Operator layouts (gfx1250 WMMA)
        if ARCH_NAME == "gfx1250":
            assert NUM_WARPS == 2 or NUM_WARPS == 4 or NUM_WARPS == 8

            if NUM_WARPS == 2:
                warp_bases = [[1, 0]]
            elif NUM_WARPS == 4:
                warp_bases = [[1, 0], [2, 0]]
            else:
                warp_bases = [[1, 0], [2, 0], [4, 0]]
            FP8_K_DIM_QK = 128 if HEAD_SIZE > 64 else 64
            # FP8_K_DIM_QK = 64
            self.qk_layout = gl.constexpr(
                gl.amd.AMDWMMALayout(
                    version=3,
                    transposed=True,
                    instr_shape=[16, 16, 32] if not FP8_DOT else [16, 16, FP8_K_DIM_QK],
                    warp_bases=warp_bases,
                )
            )
            FP8_K_DIM_PV = 128 if TILE_SIZE > 64 else 64
            # FP8_K_DIM_PV = 64
            self.pv_layout = gl.constexpr(
                gl.amd.AMDWMMALayout(
                    version=3,
                    transposed=True,
                    instr_shape=[16, 16, 32] if not FP8_DOT else [16, 16, FP8_K_DIM_PV],
                    warp_bases=warp_bases,
                )
            )

        else:
            self.qk_layout = gl.constexpr(
                gl.amd.AMDMFMALayout(
                    version=4,
                    transposed=True,
                    instr_shape=[32, 32, 16] if not FP8_DOT else [32, 32, 64],
                    warps_per_cta=[NUM_WARPS, 1],
                )
            )

            self.pv_layout = gl.constexpr(
                gl.amd.AMDMFMALayout(
                    version=4,
                    transposed=True,
                    instr_shape=[32, 32, 16] if not FP8_DOT else [32, 32, 64],
                    warps_per_cta=[NUM_WARPS, 1],
                )
            )
        # Dot operand layouts
        self.q_layout = gl.constexpr(
            gl.DotOperandLayout(0, self.qk_layout, self.K_WIDTH)
        )
        self.k_layout = gl.constexpr(
            gl.DotOperandLayout(1, self.qk_layout, self.K_WIDTH)
        )
        self.v_layout = gl.constexpr(
            gl.DotOperandLayout(1, self.pv_layout, self.K_WIDTH)
        )
        self.p_layout = gl.constexpr(
            gl.DotOperandLayout(0, self.pv_layout, self.K_WIDTH)
        )

        # Blocked layouts for global-to-shared memory loads
        ELEMENT_SIZE = 8 if Q_FP8 else 16
        MAX_LOAD = 128
        SIZE_PER_THREAD = MAX_LOAD // ELEMENT_SIZE
        HEAD_SIZE_DIV = HEAD_SIZE // 8
        self.blocked_q = gl.constexpr(
            gl.BlockedLayout(
                size_per_thread=[1, SIZE_PER_THREAD],
                threads_per_warp=[self.WARP_SIZE // 8, 8],
                warps_per_cta=[NUM_WARPS, 1],
                order=[1, 0],
            )
        )
        self.Q_CACHE_MODIFIER = gl.constexpr(".cg")
        self.KV_CACHE_MODIFIER = gl.constexpr(".cg") if ALL_DECODE else gl.constexpr("")


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
    # query_pos_qk: gl.tensor
    query_mask_qk: gl.tensor
    # context_len: gl.tensor
    context_len_q_pos_qk: gl.tensor
    QK_scale: gl.tensor
    out_scale: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q,
        key_cache_ptr,
        value_cache_ptr,
        output_ptr,
        tile_start,
        tile_end,
        safe_tile_end,
        query_mask_qk,
        context_len_q_pos_qk,
        QK_scale,
        out_scale,
    ):
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
        self.QK_scale = QK_scale
        self.out_scale = out_scale

    @gluon.jit
    def initialize(
        cfg,
        q,
        key_cache_ptr,
        value_cache_ptr,
        output_ptr,
        q_descale_ptr,
        k_descale_ptr,
        v_descale_ptr,
        out_scale_ptr,
        max_seq_prefix_len,
        q_block_local_idx,
        cur_batch_query_len,
        context_len,
        query_pos,
        query_mask,
    ):
        # Calculate tile range
        num_tiles = (max_seq_prefix_len + cfg.TILE_SIZE - 1) // cfg.TILE_SIZE
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
            tile_start = gl.maximum(0, first_allowed_key // cfg.TILE_SIZE)
            tile_end = gl.minimum((last_allowed_key // cfg.TILE_SIZE) + 1, num_tiles)

        query_pos_qk = gl.convert_layout(query_pos, gl.SliceLayout(1, cfg.qk_layout))[
            :, None
        ]
        query_mask_qk = gl.convert_layout(query_mask, cfg.qk_layout)

        context_len_q_pos_qk = context_len + query_pos_qk

        # Compute the tile index beyond which causal masking is needed.
        # min causal pos = context_len + first query pos in block
        # Tiles j < safe_tile_end have all KV positions within causal range
        # for every query row, so apply_mask_qk can be skipped.
        min_causal_pos = context_len + q_block_local_idx * cfg.BLOCK_Q
        safe_tile_end = (min_causal_pos + 1) // cfg.TILE_SIZE
        safe_tile_end = gl.minimum(safe_tile_end, tile_end)
        safe_tile_end = gl.maximum(safe_tile_end, tile_start)

        QK_scale = cfg.RCP_LN2 * cfg.SOFTMAX_SCALE

        if q_descale_ptr is not None:
            QK_scale = QK_scale * gl.load(q_descale_ptr)
        if k_descale_ptr is not None:
            QK_scale = QK_scale * gl.load(k_descale_ptr)

        if out_scale_ptr is not None:
            out_scale = 1.0 / gl.load(out_scale_ptr)
        else:
            out_scale = 1.0
        if v_descale_ptr is not None:
            out_scale = out_scale * gl.load(v_descale_ptr)

        return AttentionProgram(
            cfg,
            q,
            key_cache_ptr,
            value_cache_ptr,
            output_ptr,
            tile_start,
            tile_end,
            safe_tile_end,
            query_mask_qk,
            context_len_q_pos_qk,
            QK_scale,
            out_scale,
        )

    @gluon.jit
    def load_q_from_global(
        self,
        query_ptr,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        kv_head_idx,
        cur_batch_query_len,
        query_stride_0,
        query_stride_1,
    ):
        """Load Q from global memory."""
        offs_m = gl.arange(
            0, self.cfg.BLOCK_M, layout=gl.SliceLayout(1, self.cfg.q_layout)
        )
        offs_d = gl.arange(
            0, self.cfg.HEAD_SIZE, layout=gl.SliceLayout(0, self.cfg.q_layout)
        )
        query_pos = (
            q_block_local_idx * self.cfg.BLOCK_Q + offs_m // self.cfg.NUM_QUERIES_PER_KV
        )

        query_offset_0 = cur_batch_in_all_start_index + query_pos
        query_offset_1 = (
            kv_head_idx * self.cfg.NUM_QUERIES_PER_KV
            + offs_m % self.cfg.NUM_QUERIES_PER_KV
        )

        query_mask_0 = query_pos < cur_batch_query_len
        query_mask_1 = query_offset_1 < self.cfg.NUM_QUERY_HEADS
        query_mask = query_mask_0[:, None] & query_mask_1[:, None]

        q_offs = (
            query_offset_0[:, None] * query_stride_0
            + query_offset_1[:, None] * query_stride_1
            + offs_d[None, :]
        )
        if self.cfg.USE_STORE_BUFFER_OP:
            q = gl.amd.cdna4.buffer_load(
                query_ptr + q_offs,
                mask=query_mask,
                other=0.0,
                cache_modifier=self.cfg.Q_CACHE_MODIFIER,
            )
        else:
            q = gl.load(
                query_ptr + q_offs,
                mask=query_mask,
                other=0.0,
                cache_modifier=self.cfg.Q_CACHE_MODIFIER,
            )
        return q, query_pos, query_mask

    @gluon.jit
    def compute_qk(self, k):
        S = gl.zeros(
            [self.cfg.BLOCK_M, self.cfg.TILE_SIZE],
            dtype=gl.float32,
            layout=self.cfg.qk_layout,
        )
        if self.cfg.ARCH_NAME == "gfx1250":
            return gl.amd.gfx1250.wmma(self.q, k, S) * self.QK_scale
        else:
            return gl.amd.cdna4.mfma(self.q, k, S) * self.QK_scale

    @gluon.jit
    def apply_mask_qk(self, S, j):
        seq_offset = (
            j * self.cfg.TILE_SIZE
            + gl.arange(
                0, self.cfg.TILE_SIZE, layout=gl.SliceLayout(0, self.cfg.qk_layout)
            )[None, :]
        )

        seq_mask = seq_offset <= self.context_len_q_pos_qk
        if self.cfg.SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (self.context_len_q_pos_qk - seq_offset) < self.cfg.SLIDING_WINDOW
            )
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

    # @gluon.jit
    # def softmax_part0(self, S, M):
    #     m_ij = gl.maximum(M, gl.max(S, axis=1))
    #     m_ij = gl.where(m_ij > float("-inf"), m_ij, 0.0)
    #     m_ij_scaled = m_ij * self.QK_scale
    #     q_shifted = S * self.QK_scale - m_ij_scaled[:, None]
    #     p = gl.exp2(q_shifted)
    #     m_diff_scaled = M * self.QK_scale - m_ij_scaled
    #     alpha = gl.exp2(m_diff_scaled)
    #     return p, alpha, m_ij

    @gluon.jit
    def softmax_part1(self, p, L, acc, alpha, target_dtype=gl.bfloat16):
        l_ij = gl.sum(p, 1)
        acc = acc * alpha[:, None]
        if target_dtype != gl.bfloat16:
            p = p.to(target_dtype)
        else:
            p = p.to(target_dtype, fp_downcast_rounding="rtz")
        L = L * alpha + l_ij
        return p, L, acc

    @gluon.jit
    def compute_pv(self, p, v, acc):
        p = gl.convert_layout(p, self.cfg.p_layout, assert_trivial=True)
        if self.cfg.ARCH_NAME == "gfx1250":
            return gl.amd.gfx1250.wmma(p, v, acc)
        else:
            return gl.amd.cdna4.mfma(p, v, acc)

    @gluon.jit
    def store_output(
        self,
        out,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        kv_head_idx,
        cur_batch_query_len,
        output_stride_0,
        output_stride_1,
    ):
        offs_m_out = gl.arange(
            0, self.cfg.BLOCK_M, layout=gl.SliceLayout(1, self.cfg.blocked_q)
        )
        offs_d_out = gl.arange(
            0, self.cfg.HEAD_SIZE, layout=gl.SliceLayout(0, self.cfg.blocked_q)
        )

        query_pos_out = (
            q_block_local_idx * self.cfg.BLOCK_Q
            + offs_m_out // self.cfg.NUM_QUERIES_PER_KV
        )
        query_offset_0_out = cur_batch_in_all_start_index + query_pos_out
        query_offset_1_out = (
            kv_head_idx * self.cfg.NUM_QUERIES_PER_KV
            + offs_m_out % self.cfg.NUM_QUERIES_PER_KV
        )

        o_offs = (
            query_offset_0_out[:, None] * output_stride_0
            + query_offset_1_out[:, None] * output_stride_1
            + offs_d_out[None, :]
        )

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
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    query_start_len_ptr,  # [num_seqs+1]
    query_stride_0,
    query_stride_1,
    output_stride_0,
    output_stride_1,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_ptr,
    out_scale_ptr,
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
    TILE_SIZE: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_M: gl.constexpr,
    ARCH_NAME: gl.constexpr,
    USE_LOAD_BUFFER_OP: gl.constexpr = False,
    USE_STORE_BUFFER_OP: gl.constexpr = False,
    ALL_DECODE: gl.constexpr = False,
    USE_TDM: gl.constexpr = False,
    SHUFFLED_KV_CACHE: gl.constexpr = False,
    FP8_MIN: gl.constexpr = float8_info.min,
    FP8_MAX: gl.constexpr = float8_info.max,
):
    NUM_WARPS: gl.constexpr = gl.num_warps()
    # Workgroup offsets
    kv_head_idx = gl.program_id(0)
    q_block_global_idx = gl.num_programs(1) - 1 - gl.program_id(1)
    # Q dtype determines the dot product dtype
    # KV gets casted to Q dtype for dot product
    Q_FP8: gl.constexpr = query_ptr.dtype.is_fp8()
    KV_FP8: gl.constexpr = key_cache_ptr.dtype.is_fp8()
    # Build config with all layouts and derived constants
    cfg = AttentionConfig(
        ARCH_NAME,
        NUM_WARPS,
        HEAD_SIZE,
        BLOCK_SIZE,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        SLIDING_WINDOW,
        SCALE,
        USE_LOAD_BUFFER_OP,
        USE_STORE_BUFFER_OP,
        ALL_DECODE,
        SHUFFLED_KV_CACHE,
        Q_FP8,
        KV_FP8,
    )

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
    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, cfg.BLOCK_Q
    )

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
    query_offset_1 = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_m % cfg.NUM_QUERIES_PER_KV
    )

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < NUM_QUERY_HEADS
    query_mask = query_mask_0[:, None] & query_mask_1[:, None]

    q_offs = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    q = gl.amd.cdna4.buffer_load(
        ptr=query_ptr,
        offsets=q_offs,
        mask=query_mask,
        other=0.0,
        cache=cfg.Q_CACHE_MODIFIER,
    )

    seq_len = gl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len
    block_tables_ptr_shifted = block_tables_ptr + seq_idx * block_table_stride

    # Max KV position that any query in this block attends to
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * cfg.BLOCK_Q
        + (BLOCK_M - 1) // cfg.NUM_QUERIES_PER_KV
        + 1
    )
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)

    # build program
    pgm = AttentionProgram.initialize(
        cfg,
        q,
        key_cache_ptr,
        value_cache_ptr,
        output_ptr,
        q_descale_ptr,
        k_descale_ptr,
        v_descale_ptr,
        out_scale_ptr,
        max_seq_prefix_len,
        q_block_local_idx,
        cur_batch_query_len,
        context_len,
        query_pos,
        query_mask,
    )
    if USE_TDM:
        if TILE_SIZE == BLOCK_SIZE:
            KVLoader: gl.constexpr = TDMKVLoader
        else:
            KVLoader: gl.constexpr = TDMGatherKVLoader
    else:
        gl.static_assert(
            TILE_SIZE == BLOCK_SIZE,
            "With async kv loader, TILE_SIZE must be equal to BLOCK_SIZE",
        )
        KVLoader: gl.constexpr = AsyncKVLoader

    kv_loader = KVLoader.initialize(
        cfg,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr_shifted,
        kv_head_idx,
        num_blocks,
        stride_k_cache_0,
        stride_k_cache_1,
        stride_k_cache_2,
        stride_k_cache_3,
        stride_v_cache_0,
        stride_v_cache_1,
        stride_v_cache_2,
        stride_v_cache_3,
    )

    # Initialize accumulators
    if not USE_SINKS:
        M = gl.full(
            [BLOCK_M],
            float("-inf"),
            dtype=gl.float32,
            layout=gl.SliceLayout(1, cfg.pv_layout),
        )
    else:
        offs_m_pv = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout))
        query_offset_1_pv = (
            kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_m_pv % cfg.NUM_QUERIES_PER_KV
        )
        query_mask_1_pv = query_offset_1_pv < NUM_QUERY_HEADS
        # Using regular approach: Prescale with RCP_LN2, needed for exp2
        # FMA based approach: Prescale with / SCALE
        M = (
            gl.load(
                sink_ptr + query_offset_1_pv,
                mask=query_mask_1_pv,
                other=float("-inf"),
            ).to(dtype=gl.float32)
            * cfg.RCP_LN2
        )

    L = gl.full(
        [BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, cfg.pv_layout)
    )
    acc = gl.zeros([BLOCK_M, HEAD_SIZE], dtype=gl.float32, layout=cfg.pv_layout)
    # TODO (cagri): Assuming stride_k_cache_0 == stride_v_cache_0
    # Prologue: load first tile's block index and issue async K, V loads
    physical_block_idx = kv_loader.load_block_ids(pgm.tile_start)

    # rotating buffer index logic
    # TODO (cagri): Loop unrolling can get rid of this
    buffer_id: gl.int32 = 0
    kv_loader.load_k_to_shared(physical_block_idx, buffer_id=buffer_id)
    kv_loader.load_v_to_shared(physical_block_idx, buffer_id=buffer_id)
    # Main attention loop over KV tiles (staged, num_stages=2)
    for j in range(pgm.tile_start, pgm.safe_tile_end):

        next_physical_block_idx = kv_loader.load_block_ids(j + 1)
        k = kv_loader.load_k_from_shared(
            wait_count=1, target_dtype=q.dtype, buffer_id=buffer_id
        )
        # Prefetch next tile (shared is free since k, v are in registers)
        kv_loader.load_k_to_shared(next_physical_block_idx, buffer_id=1 - buffer_id)
        kv_loader.load_v_to_shared(next_physical_block_idx, buffer_id=1 - buffer_id)

        # Compute attention for current tile
        S = pgm.compute_qk(k)
        if SLIDING_WINDOW > 0:
            S = pgm.apply_mask_qk(S, j)
        S = gl.convert_layout(S, pgm.cfg.pv_layout, assert_trivial=True)
        p, alpha, M = pgm.softmax_part0(S, M)
        p, L, acc = pgm.softmax_part1(p, L, acc, alpha, target_dtype=q.dtype)

        v = kv_loader.load_v_from_shared(
            wait_count=2, target_dtype=q.dtype, buffer_id=buffer_id
        )
        acc = pgm.compute_pv(p, v, acc)
        buffer_id = 1 - buffer_id

    for j in range(pgm.safe_tile_end, pgm.tile_end - 1):
        # with gl.amd.warp_pipeline_stage("stage0", priority=2):
        next_physical_block_idx = kv_loader.load_block_ids(j + 1)
        k = kv_loader.load_k_from_shared(
            wait_count=1, target_dtype=q.dtype, buffer_id=buffer_id
        )
        # Prefetch next tile (shared is free since k, v are in registers)
        kv_loader.load_k_to_shared(next_physical_block_idx, buffer_id=1 - buffer_id)
        kv_loader.load_v_to_shared(next_physical_block_idx, buffer_id=1 - buffer_id)
        # Compute attention for current tile
        S = pgm.compute_qk(k)
        S = pgm.apply_mask_qk(S, j)
        S = gl.convert_layout(S, pgm.cfg.pv_layout, assert_trivial=True)
        p, alpha, M = pgm.softmax_part0(S, M)
        p, L, acc = pgm.softmax_part1(p, L, acc, alpha, target_dtype=k.dtype)
        # with gl.amd.warp_pipeline_stage("stage2", priority=0):
        v = kv_loader.load_v_from_shared(
            wait_count=2, target_dtype=q.dtype, buffer_id=buffer_id
        )
        acc = pgm.compute_pv(p, v, acc)
        buffer_id = 1 - buffer_id

    # Load k_i, v_i from shared into registers
    k = kv_loader.load_k_from_shared(
        wait_count=1, target_dtype=q.dtype, buffer_id=buffer_id
    )
    # Compute attention for current tile
    S = pgm.compute_qk(k)
    S = pgm.apply_mask_qk(S, pgm.tile_end - 1)
    S = gl.convert_layout(S, pgm.cfg.pv_layout, assert_trivial=True)
    p, alpha, M = pgm.softmax_part0(S, M)
    p, L, acc = pgm.softmax_part1(p, L, acc, alpha, target_dtype=k.dtype)
    v = kv_loader.load_v_from_shared(
        wait_count=0, target_dtype=q.dtype, buffer_id=buffer_id
    )
    acc = pgm.compute_pv(p, v, acc)
    # Normalize and store output
    l_recip = pgm.out_scale / L[:, None]
    acc = acc * l_recip
    if output_ptr.dtype.is_fp8():
        # clamp to FP8 range
        acc = gl.minimum(acc, FP8_MAX)
        acc = gl.maximum(acc, FP8_MIN)

    pgm.store_output(
        acc,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        kv_head_idx,
        cur_batch_query_len,
        output_stride_0,
        output_stride_1,
    )


def unified_attention(
    q,
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
    sinks,
    output_scale=None,
    new_kv_layout=False,
    num_kv_blocks=1,
    use_tdm=True,
    waves_per_eu=4,
    shuffled_kv_cache=False,
    num_warps=4,
    block_m=128,
):
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
        output_scale: Output scale
        sinks: Sinks tensor [num_query_heads,]
    """
    NUM_SEQS = len(seqused_k)
    NUM_Q_HEADS = q.shape[1]
    HEAD_SIZE = q.shape[2]
    num_blocks = k.shape[0]
    if shuffled_kv_cache:
        # key_cache: num_blocks, num_kv_heads, block_size // 16, head_size * 16
        # value_cache: num_blocks, num_kv_heads, head_size // 16, block_size * 16
        _, NUM_KV_HEADS, block_size, _ = k.shape
        BLOCK_SIZE = block_size * 16

    else:
        if new_kv_layout:
            assert use_tdm, "With new kv layout, USE_TDM must be True"
            assert (
                not shuffled_kv_cache
            ), "With new kv layout, SHUFFLED_KV_CACHE must be False"
            BLOCK_SIZE = k.shape[2]
            NUM_KV_HEADS = k.shape[1]
        else:
            # key_cache: num_blocks, num_kv_heads, block_size, head_size
            # value_cache: num_blocks, num_kv_heads, block_size, head_size
            assert (
                num_kv_blocks == 1
            ), "With original kv layout, num_kv_blocks must be 1"
            if not use_tdm:
                assert (
                    not shuffled_kv_cache
                ), "Shuffling is only supported with TDM, without TDM-Gather"
            BLOCK_SIZE = k.shape[1]
            NUM_KV_HEADS = k.shape[2]

    # if use_tdm:
    #     assert ARCH_NAME == "gfx1250", "With TDM, ARCH must be gfx1250"
    BLOCK_M = block_m
    SLIDING_WINDOW = 1 + window_size[0]
    ALL_DECODE = max_seqlen_q == 1
    NUM_QUERIES_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
    BLOCK_Q = BLOCK_M // NUM_QUERIES_PER_KV
    total_query_blocks = q.shape[0] // BLOCK_Q + NUM_SEQS
    assert (
        num_kv_blocks & (num_kv_blocks - 1) == 0
    ), "num_kv_blocks must be a power of 2"
    TILE_SIZE = num_kv_blocks * BLOCK_SIZE
    ARCH_NAME = arch_info.get_arch()
    NUM_WARPS = num_warps
    kv_size = k.nelement() * k.element_size()
    MAX_INT32 = 2**31 - 1
    USE_LOAD_BUFFER_OP = ARCH_NAME != "gfx1250" and kv_size <= MAX_INT32
    USE_STORE_BUFFER_OP = out.nelement() * out.element_size() <= MAX_INT32
    Q_FP8 = q.element_size() == 1
    KV_FP8 = k.element_size() == 1
    # waves_per_eu = 2 if HEAD_SIZE < 128 else 2
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
        k_descale_ptr=k_descale,
        v_descale_ptr=v_descale,
        q_descale_ptr=q_descale,
        out_scale_ptr=output_scale,
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
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        ARCH_NAME=ARCH_NAME,
        waves_per_eu=waves_per_eu,
        USE_LOAD_BUFFER_OP=USE_LOAD_BUFFER_OP,
        USE_STORE_BUFFER_OP=USE_STORE_BUFFER_OP,
        num_warps=NUM_WARPS,
        ALL_DECODE=ALL_DECODE,
        USE_TDM=use_tdm,
        SHUFFLED_KV_CACHE=shuffled_kv_cache,
    )

    if PRINT_IRS and getattr(unified_attention, "print", False) == False:
        setattr(unified_attention, "print", True)
        print_irs_to_files(
            attn_kernel,
            f"unified_attention_2d_gluon_num_warps_{NUM_WARPS}_block_m_{BLOCK_M}_tile_size_{TILE_SIZE}_block_size_{BLOCK_SIZE}_head_size_{HEAD_SIZE}_sfl_{shuffled_kv_cache}",
        )
    return attn_kernel


def print_irs_to_files(compiled_kernel, prefix):
    for key in compiled_kernel.asm.keys():
        with open(f"{prefix}_{key}.txt", "w") as fptr:
            print(compiled_kernel.asm[key], file=fptr)
