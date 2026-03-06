# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
from re import T
import triton
import triton.language as tl
import torch
from aiter.ops.triton.utils.types import e4m3_dtype
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import aiter.ops.triton.utils._triton.arch_info as arch_info
from triton.language.core import _aggregate as aggregate
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

import math

# from triton._C.libtriton.gluon_ir import make_cga_layout

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
MMA_operation: gl.constexpr = (
    gl.amd.gfx1250.wmma if gl.constexpr(IS_DEVICE_ARCH_GFX12) else gl.amd.cdna4.mfma
)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64
WAPR_SIZE_LOG2 = int(math.log2(WARP_SIZE))

float8_info = torch.finfo(e4m3_dtype)


@gluon.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.math.exp2(Sdiv)
    p2 = tl.math.exp2(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@aggregate
class AttentionConfig:
    """Configuration for unified attention layouts and derived constants."""

    # Core dimensions
    HEAD_SIZE: gl.constexpr
    BLOCK_SIZE: gl.constexpr
    NUM_BLOCKS_GATHER_PER_TILE: gl.constexpr
    NUM_SEGMENTS_PER_SEQ: gl.constexpr
    BLOCK_M: gl.constexpr
    NUM_QUERY_HEADS: gl.constexpr
    NUM_KV_HEADS: gl.constexpr
    SLIDING_WINDOW: gl.constexpr

    # Derived constants
    TILE_SIZE: gl.constexpr
    NUM_QUERIES_PER_KV: gl.constexpr
    BLOCK_Q: gl.constexpr
    RCP_LN2: gl.constexpr
    QK_SCALE: gl.constexpr

    # Operator layouts (CDNA4 MFMA)
    QK_WMMA_LAYOUT: gl.constexpr
    PV_WMMA_LAYOUT: gl.constexpr

    # Dot operand layouts
    Q_DOT_LAYOUT: gl.constexpr
    K_DOT_LAYOUT: gl.constexpr
    V_DOT_LAYOUT: gl.constexpr
    P_DOT_LAYOUT: gl.constexpr

    # Layout for loading Q
    Q_LOAD_LAYOUT: gl.constexpr

    # Shared memory layouts
    Q_SHARED_LAYOUT: gl.constexpr
    K_SHARED_LAYOUT: gl.constexpr
    V_SHARED_LAYOUT: gl.constexpr
    GATHER_BLOCKED_LAYOUT: gl.constexpr
    K_LOAD_LAYOUT: gl.constexpr
    V_LOAD_LAYOUT: gl.constexpr

    q_cache_modifier: gl.constexpr
    kv_cache_modifier: gl.constexpr

    USE_ALIBI_SLOPES: gl.constexpr
    USE_QQ_BIAS: gl.constexpr
    USE_SOFTCAP: gl.constexpr
    USE_SINKS: gl.constexpr
    USE_LOAD_BUFFER_OP: gl.constexpr
    USE_STORE_BUFFER_OP: gl.constexpr

    NUM_STAGES: gl.constexpr
    SHUFFLED_KV_CACHE: gl.constexpr
    IS_Q_FP8: gl.constexpr
    IS_KV_FP8: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        HEAD_SIZE,
        BLOCK_SIZE,
        NUM_BLOCKS_GATHER_PER_TILE,
        NUM_SEGMENTS_PER_SEQ,
        BLOCK_M,
        BLOCK_Q,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        SLIDING_WINDOW,
        NUM_WARPS,
        WARP_SIZE,
        NUM_STAGES,
        SCALE,
        USE_ALIBI_SLOPES,
        USE_QQ_BIAS,
        USE_SOFTCAP,
        USE_SINKS,
        USE_LOAD_BUFFER_OP,
        USE_STORE_BUFFER_OP,
        SHUFFLED_KV_CACHE,
        IS_Q_FP8,
        IS_KV_FP8,
    ):
        # Constants
        self.HEAD_SIZE = gl.constexpr(HEAD_SIZE)
        self.BLOCK_SIZE = gl.constexpr(BLOCK_SIZE)
        self.NUM_BLOCKS_GATHER_PER_TILE = gl.constexpr(NUM_BLOCKS_GATHER_PER_TILE)
        self.NUM_SEGMENTS_PER_SEQ = gl.constexpr(NUM_SEGMENTS_PER_SEQ)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.NUM_QUERY_HEADS = gl.constexpr(NUM_QUERY_HEADS)
        self.NUM_KV_HEADS = gl.constexpr(NUM_KV_HEADS)
        self.SLIDING_WINDOW = gl.constexpr(SLIDING_WINDOW)
        self.NUM_STAGES = gl.constexpr(NUM_STAGES)
        self.SHUFFLED_KV_CACHE = gl.constexpr(SHUFFLED_KV_CACHE)
        self.IS_Q_FP8 = gl.constexpr(IS_Q_FP8)
        self.IS_KV_FP8 = gl.constexpr(IS_KV_FP8)
        # Derived constants
        self.TILE_SIZE = gl.constexpr(BLOCK_SIZE * NUM_BLOCKS_GATHER_PER_TILE)
        self.NUM_QUERIES_PER_KV = gl.constexpr(NUM_QUERY_HEADS // NUM_KV_HEADS)
        self.BLOCK_Q = gl.constexpr(BLOCK_Q)
        self.RCP_LN2 = gl.constexpr(1.4426950408889634)
        self.QK_SCALE = gl.constexpr(SCALE * self.RCP_LN2)
        self.USE_ALIBI_SLOPES = gl.constexpr(USE_ALIBI_SLOPES)
        self.USE_QQ_BIAS = gl.constexpr(USE_QQ_BIAS)
        self.USE_SOFTCAP = gl.constexpr(USE_SOFTCAP)
        self.USE_SINKS = gl.constexpr(USE_SINKS)
        self.USE_LOAD_BUFFER_OP = gl.constexpr(USE_LOAD_BUFFER_OP)
        self.USE_STORE_BUFFER_OP = gl.constexpr(USE_STORE_BUFFER_OP)

        # gl.static_assert(NUM_WARPS == 2 or NUM_WARPS == 4, "NUM_WARPS must be 2 or 4")
        assert NUM_WARPS == 1 or NUM_WARPS == 2 or NUM_WARPS == 4

        if NUM_WARPS == 1:
            warp_bases_qk = []
            warp_bases_pv = []
        elif NUM_WARPS == 2:
            warp_bases_qk = [(1, 0)]
            warp_bases_pv = [(0, 1)]
        elif NUM_WARPS == 4:
            warp_bases_qk = [(1, 0), (2, 0)]
            warp_bases_pv = [(0, 1), (0, 2)]

        # gl.static_assert(
        #     WARP_SIZE == 32 or WARP_SIZE == 64, "WARP_SIZE must be 32 or 64"
        # )
        assert WARP_SIZE == 32

        self.QK_WMMA_LAYOUT = gl.constexpr(
            gl.amd.AMDWMMALayout(
                version=3,
                transposed=True,
                warp_bases=warp_bases_qk,
                reg_bases=[],
                instr_shape=[16, 16, 32 if not self.IS_Q_FP8 else 64],
            )
        )

        self.PV_WMMA_LAYOUT = gl.constexpr(
            gl.amd.AMDWMMALayout(
                version=3,
                transposed=True,
                warp_bases=warp_bases_pv,
                reg_bases=[],
                instr_shape=[16, 16, 32 if not self.IS_KV_FP8 else 64],
            )
        )
        self.Q_DOT_LAYOUT = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=self.QK_WMMA_LAYOUT, k_width=8)
        )
        self.K_DOT_LAYOUT = gl.constexpr(
            gl.DotOperandLayout(operand_index=1, parent=self.QK_WMMA_LAYOUT, k_width=8)
        )
        self.P_DOT_LAYOUT = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=self.PV_WMMA_LAYOUT, k_width=8)
        )
        self.V_DOT_LAYOUT = gl.constexpr(
            gl.DotOperandLayout(operand_index=1, parent=self.PV_WMMA_LAYOUT, k_width=8)
        )

        # gl.static_assert(
        #     NUM_BLOCKS_GATHER_PER_TILE == 1
        #     or NUM_BLOCKS_GATHER_PER_TILE == 4
        #     or NUM_BLOCKS_GATHER_PER_TILE == 8,
        #     "NUM_BLOCKS_GATHER_PER_TILE must be 1, 4, or 8",
        # )
        assert (
            NUM_BLOCKS_GATHER_PER_TILE == 1
            or NUM_BLOCKS_GATHER_PER_TILE == 4
            or NUM_BLOCKS_GATHER_PER_TILE == 8
        )

        self.Q_SHARED_LAYOUT = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for(
                interval_padding_pairs=[[HEAD_SIZE, 8]],
                shape=[BLOCK_M, HEAD_SIZE],
                order=[1, 0],
            )
        )

        if self.SHUFFLED_KV_CACHE:
            self.K_SHARED_LAYOUT = gl.constexpr(
                gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
            )
            self.V_SHARED_LAYOUT = gl.constexpr(
                gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
            )
            if NUM_BLOCKS_GATHER_PER_TILE == 1:
                self.GATHER_BLOCKED_LAYOUT = gl.constexpr(None)
            else:
                self.GATHER_BLOCKED_LAYOUT = gl.constexpr(
                    gl.BlockedLayout(
                        size_per_thread=[NUM_BLOCKS_GATHER_PER_TILE],
                        threads_per_warp=[WARP_SIZE],
                        warps_per_cta=[NUM_WARPS],
                        order=[0],
                    )
                )
        elif NUM_BLOCKS_GATHER_PER_TILE == 1:
            self.K_SHARED_LAYOUT = gl.constexpr(
                gl.PaddedSharedLayout.with_identity_for(
                    interval_padding_pairs=[[HEAD_SIZE, 8]],
                    shape=([BLOCK_SIZE, HEAD_SIZE]),
                    order=[1, 0],
                )
            )
            self.V_SHARED_LAYOUT = gl.constexpr(
                gl.PaddedSharedLayout.with_identity_for(
                    interval_padding_pairs=[[HEAD_SIZE, 8]],
                    shape=[BLOCK_SIZE, HEAD_SIZE],
                    order=[1, 0],
                )
            )
            self.GATHER_BLOCKED_LAYOUT = gl.constexpr(None)
        else:
            self.K_SHARED_LAYOUT = gl.constexpr(
                gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
            )
            self.V_SHARED_LAYOUT = gl.constexpr(
                gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
            )
            # TODO Disabled PaddedSharedLayout for now
            # self.K_SHARED_LAYOUT = gl.constexpr(
            #     gl.PaddedSharedLayout.with_identity_for(
            #         interval_padding_pairs=[[BLOCK_SIZE * HEAD_SIZE, 8]],
            #         shape=([NUM_BLOCKS_GATHER_PER_TILE, BLOCK_SIZE * HEAD_SIZE]),
            #         order=[1, 0],
            #     )
            # )
            # self.V_SHARED_LAYOUT = gl.constexpr(
            #     gl.PaddedSharedLayout.with_identity_for(
            #         interval_padding_pairs=[[BLOCK_SIZE * HEAD_SIZE, 8]],
            #         shape=[NUM_BLOCKS_GATHER_PER_TILE, BLOCK_SIZE * HEAD_SIZE],
            #         order=[1, 0],
            #     )
            # )
            self.GATHER_BLOCKED_LAYOUT = gl.constexpr(
                gl.BlockedLayout(
                    size_per_thread=[NUM_BLOCKS_GATHER_PER_TILE],
                    threads_per_warp=[WARP_SIZE],
                    warps_per_cta=[NUM_WARPS],
                    order=[0],
                )
            )

        # size_per_thread along the fastest moving dimension is set to 8 (BF16)
        size_per_thread_fastest_dim = gl.constexpr(8)
        # size_per_thread * threads_per_warp along the fastest moving dimension is set to HEAD_SIZE with only 1 warp_per_cta,
        # therefore, threads_per_warp along the fastest moving dimension should be HEAD_SIZE // size_per_thread_fastest_dim
        # clamp the threads_per_warp along the fastest moving dimension to 1 ~ WARP_SIZE
        threads_per_warp_fastest_dim = max(
            min((HEAD_SIZE // size_per_thread_fastest_dim), WARP_SIZE), 1
        )

        self.Q_LOAD_LAYOUT = gl.constexpr(
            gl.BlockedLayout(
                size_per_thread=[1, size_per_thread_fastest_dim],
                threads_per_warp=[
                    WARP_SIZE // threads_per_warp_fastest_dim,
                    threads_per_warp_fastest_dim,
                ],
                warps_per_cta=[NUM_WARPS, 1],
                order=[1, 0],
            )
        )
        if self.SHUFFLED_KV_CACHE:
            if self.NUM_BLOCKS_GATHER_PER_TILE == 1:
                # self.K_LOAD_LAYOUT = self.make_kv_cache_shuffled_layout(
                #     self.TILE_SIZE // 16,
                #     self.HEAD_SIZE * 16,
                #     1,
                # )
                # self.V_LOAD_LAYOUT = self.make_kv_cache_shuffled_layout(
                #     self.HEAD_SIZE // 16,
                #     self.TILE_SIZE * 16,
                #     NUM_WARPS,
                # )
                self.K_LOAD_LAYOUT = gl.constexpr(None)
                self.V_LOAD_LAYOUT = gl.constexpr(None)
            else:
                self.K_LOAD_LAYOUT = gl.constexpr(None)
                self.V_LOAD_LAYOUT = gl.constexpr(None)
        else:
            self.K_LOAD_LAYOUT = gl.constexpr(None)
            self.V_LOAD_LAYOUT = gl.constexpr(None)

        self.q_cache_modifier = gl.constexpr(".cg")
        self.kv_cache_modifier = gl.constexpr(".cg")

    @gluon.constexpr_function
    def make_kv_cache_shuffled_layout(
        self,
        BLOCK_SIZE_N_SHFL,
        BLOCK_SIZE_INNER_DIM_SHFL,
        fastest_dim_num_warps,
        dtype=torch.bfloat16,
    ):
        num_warps_log2 = int(math.log2(fastest_dim_num_warps))
        BLOCK_SIZE_N_SHFL_log2 = int(math.log2(BLOCK_SIZE_N_SHFL))
        BLOCK_SIZE_INNER_DIM_SHFL_log2 = int(math.log2(BLOCK_SIZE_INNER_DIM_SHFL))
        # TODO: support e4m3_dtype and mxfp4x2
        # assert dtype in [torch.bfloat16, e4m3_dtype, torch.uint8], f"Unsupported dtype: {dtype} for making linear layout for shuffled weights"
        assert dtype in [
            torch.bfloat16
        ], f"Unsupported dtype: {dtype} for making linear layout for shuffled weights"
        if dtype == torch.bfloat16:
            # (8 elements per thread for BF16)
            coalesced_size_log2 = 3
        elif dtype == e4m3_dtype:
            # (16 elements per thread for e4m3_dtype)
            coalesced_size_log2 = 4
        else:
            # (16*2 elements per thread for mxfp4x2)
            coalesced_size_log2 = 4
        assert (
            BLOCK_SIZE_INNER_DIM_SHFL_log2 > coalesced_size_log2 + WAPR_SIZE_LOG2
        ), "BLOCK_SIZE_INNER_DIM_SHFL_log2 must be greater than coalesced_size_log2 + WAPR_SIZE_LOG2, please increase block_size to at least 64"
        reg_bases = (
            [[0, 1 << v] for v in range(coalesced_size_log2)]
            + [
                [0, 1 << v]
                for v in range(
                    coalesced_size_log2 + WAPR_SIZE_LOG2, BLOCK_SIZE_INNER_DIM_SHFL_log2
                )
            ]
            + [
                [0, 1 << v]
                for v in range(
                    num_warps_log2 + BLOCK_SIZE_INNER_DIM_SHFL_log2,
                    BLOCK_SIZE_INNER_DIM_SHFL_log2 + BLOCK_SIZE_N_SHFL_log2,
                )
            ]
        )
        lane_bases = [
            [0, 1 << v]
            for v in range(coalesced_size_log2, coalesced_size_log2 + WAPR_SIZE_LOG2)
        ]
        if num_warps_log2 > 0:
            warp_bases = [
                [0, 1 << v]
                for v in range(
                    BLOCK_SIZE_INNER_DIM_SHFL_log2,
                    num_warps_log2 + BLOCK_SIZE_INNER_DIM_SHFL_log2,
                )
            ]
        else:
            warp_bases = [[0, 0]]

        layout = gl.constexpr(
            gl.DistributedLinearLayout(
                reg_bases=reg_bases,
                lane_bases=lane_bases,
                warp_bases=warp_bases,
                block_bases=[],
                shape=[1, BLOCK_SIZE_N_SHFL * BLOCK_SIZE_INNER_DIM_SHFL],
            )
        )
        return layout


@aggregate
class AttentionProgram:
    """Program state and core operations for the unified attention kernel."""

    cfg: AttentionConfig

    q: gl.tensor
    k_shared: gl.shared_memory_descriptor
    v_shared: gl.shared_memory_descriptor

    key_cache_ptr: gl.tensor
    value_cache_ptr: gl.tensor
    output_ptr: gl.tensor
    # segm_output_ptr: gl.tensor
    segm_max_ptr: gl.tensor
    segm_expsum_ptr: gl.tensor

    tile_start: gl.tensor
    tile_end: gl.tensor
    safe_tile_end: gl.tensor
    kv_head_idx: gl.tensor
    query_mask_qk: gl.tensor
    context_len: gl.tensor
    context_len_q_pos_qk: gl.tensor
    query_pos_qk: gl.tensor
    query_mask_qk: gl.tensor
    query_offset_0_qk: gl.tensor
    query_offset_1_qk: gl.tensor
    query_mask_0_qk: gl.tensor
    query_mask_1_qk: gl.tensor
    query_offset_0_pv: gl.tensor
    query_offset_1_pv: gl.tensor
    query_mask_0_pv: gl.tensor
    query_mask_1_pv: gl.tensor

    k_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    v_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    stride_k_cache_0: gl.tensor
    stride_k_cache_1: gl.tensor
    stride_k_cache_2: gl.tensor
    stride_k_cache_3: gl.tensor
    stride_v_cache_0: gl.tensor
    stride_v_cache_1: gl.tensor
    stride_v_cache_2: gl.tensor
    stride_v_cache_3: gl.tensor

    qq_bias_stride_0: gl.tensor
    softcap: gl.tensor
    q_scale: gl.tensor
    k_scale: gl.tensor
    v_scale: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q,
        k_shared,
        v_shared,
        key_cache_ptr,
        value_cache_ptr,
        output_ptr,
        segm_max_ptr,
        segm_expsum_ptr,
        tile_start,
        tile_end,
        safe_tile_end,
        kv_head_idx,
        context_len,
        context_len_q_pos_qk,
        query_pos_qk,
        query_mask_qk,
        query_offset_0_qk,
        query_offset_1_qk,
        query_mask_0_qk,
        query_mask_1_qk,
        query_offset_0_pv,
        query_offset_1_pv,
        query_mask_0_pv,
        query_mask_1_pv,
        k_desc,
        v_desc,
        stride_k_cache_0,
        stride_k_cache_1,
        stride_k_cache_2,
        stride_k_cache_3,
        stride_v_cache_0,
        stride_v_cache_1,
        stride_v_cache_2,
        stride_v_cache_3,
        qq_bias_stride_0,
        softcap,
        q_scale,
        k_scale,
        v_scale,
    ):
        self.cfg = cfg
        self.q = q
        self.key_cache_ptr = key_cache_ptr
        self.value_cache_ptr = value_cache_ptr
        self.output_ptr = output_ptr
        self.segm_max_ptr = segm_max_ptr
        self.segm_expsum_ptr = segm_expsum_ptr
        self.k_shared = k_shared
        self.v_shared = v_shared
        self.k_desc = k_desc
        self.v_desc = v_desc
        self.tile_start = tile_start
        self.tile_end = tile_end
        self.safe_tile_end = safe_tile_end
        self.context_len = context_len
        self.context_len_q_pos_qk = context_len_q_pos_qk
        self.query_pos_qk = query_pos_qk
        self.query_mask_qk = query_mask_qk
        self.query_offset_0_qk = query_offset_0_qk
        self.query_offset_1_qk = query_offset_1_qk
        self.query_mask_0_qk = query_mask_0_qk
        self.query_mask_1_qk = query_mask_1_qk
        self.query_offset_0_pv = query_offset_0_pv
        self.query_offset_1_pv = query_offset_1_pv
        self.query_mask_0_pv = query_mask_0_pv
        self.query_mask_1_pv = query_mask_1_pv
        self.kv_head_idx = kv_head_idx
        self.stride_k_cache_0 = stride_k_cache_0
        self.stride_k_cache_1 = stride_k_cache_1
        self.stride_k_cache_2 = stride_k_cache_2
        self.stride_k_cache_3 = stride_k_cache_3
        self.stride_v_cache_0 = stride_v_cache_0
        self.stride_v_cache_1 = stride_v_cache_1
        self.stride_v_cache_2 = stride_v_cache_2
        self.stride_v_cache_3 = stride_v_cache_3
        self.qq_bias_stride_0 = qq_bias_stride_0
        self.q_scale = q_scale
        self.k_scale = k_scale
        self.v_scale = v_scale
        self.softcap = softcap

    @gluon.jit
    def initialize(
        cfg: AttentionConfig,
        q,
        key_cache_ptr,
        value_cache_ptr,
        output_ptr,
        segm_max_ptr,
        segm_expsum_ptr,
        max_seq_prefix_len,
        q_block_local_idx,
        cur_batch_query_len,
        context_len,
        kv_head_idx,
        num_blocks,
        query_pos_qk,
        query_mask_qk,
        query_offset_0_qk,
        query_offset_1_qk,
        query_mask_0_qk,
        query_mask_1_qk,
        query_offset_0_pv,
        query_offset_1_pv,
        query_mask_0_pv,
        query_mask_1_pv,
        segm_idx,
        tiles_per_segment,
        stride_k_cache_0,
        stride_k_cache_1,
        stride_k_cache_2,
        stride_k_cache_3,
        stride_v_cache_0,
        stride_v_cache_1,
        stride_v_cache_2,
        stride_v_cache_3,
        qq_bias_stride_0,
        q_scale,
        k_scale,
        v_scale,
        softcap,
    ):
        # the last dimension of the stride should always be 1
        # gl.static_assert(stride_k_cache_3 == 1)
        # gl.static_assert(stride_v_cache_3 == 1)
        # if cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
        #     # in TDM mode, KV cache shape should be [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE]
        #     gl.static_assert(stride_k_cache_0 // stride_k_cache_1 == cfg.BLOCK_SIZE)
        #     gl.static_assert(stride_v_cache_0 // stride_v_cache_1 == cfg.BLOCK_SIZE)
        #     gl.static_assert(stride_k_cache_2 == cfg.HEAD_SIZE)
        #     gl.static_assert(stride_v_cache_2 == cfg.HEAD_SIZE)
        # else:
        #     # in TDM gather mode, KV cache shape should be [num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE]
        #     gl.static_assert(stride_k_cache_0 // stride_k_cache_1 == cfg.NUM_KV_HEADS)
        #     gl.static_assert(stride_v_cache_0 // stride_v_cache_1 == cfg.NUM_KV_HEADS)

        if cfg.SHUFFLED_KV_CACHE:
            if cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
                k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                    base=key_cache_ptr,
                    shape=(
                        num_blocks * cfg.NUM_KV_HEADS,
                        cfg.BLOCK_SIZE * cfg.HEAD_SIZE,
                    ),
                    strides=(stride_k_cache_1, 1),
                    block_shape=(gl.constexpr(1), cfg.BLOCK_SIZE * cfg.HEAD_SIZE),
                    layout=cfg.K_SHARED_LAYOUT,
                )
                v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                    base=value_cache_ptr,
                    shape=(
                        num_blocks * cfg.NUM_KV_HEADS,
                        cfg.HEAD_SIZE * cfg.BLOCK_SIZE,
                    ),
                    strides=(stride_v_cache_1, 1),
                    block_shape=(gl.constexpr(1), cfg.HEAD_SIZE * cfg.BLOCK_SIZE),
                    layout=cfg.V_SHARED_LAYOUT,
                )
            else:
                k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                    base=key_cache_ptr,
                    shape=(
                        num_blocks * cfg.NUM_KV_HEADS,
                        cfg.BLOCK_SIZE * cfg.HEAD_SIZE,
                    ),
                    strides=(stride_k_cache_1, 1),
                    block_shape=(
                        cfg.NUM_BLOCKS_GATHER_PER_TILE,
                        cfg.BLOCK_SIZE * cfg.HEAD_SIZE,
                    ),
                    layout=cfg.K_SHARED_LAYOUT,
                )
                v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                    base=value_cache_ptr,
                    shape=(
                        num_blocks * cfg.NUM_KV_HEADS,
                        cfg.HEAD_SIZE * cfg.BLOCK_SIZE,
                    ),
                    strides=(stride_v_cache_1, 1),
                    block_shape=(
                        cfg.NUM_BLOCKS_GATHER_PER_TILE,
                        cfg.HEAD_SIZE * cfg.BLOCK_SIZE,
                    ),
                    layout=cfg.V_SHARED_LAYOUT,
                )
        elif cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
            k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=key_cache_ptr,
                shape=(num_blocks * cfg.BLOCK_SIZE, cfg.NUM_KV_HEADS * cfg.HEAD_SIZE),
                strides=(stride_k_cache_1, 1),
                block_shape=(cfg.BLOCK_SIZE, cfg.HEAD_SIZE),
                layout=cfg.K_SHARED_LAYOUT,
            )
            v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=value_cache_ptr,
                shape=(num_blocks * cfg.BLOCK_SIZE, cfg.NUM_KV_HEADS * cfg.HEAD_SIZE),
                strides=(stride_v_cache_1, 1),
                block_shape=(cfg.BLOCK_SIZE, cfg.HEAD_SIZE),
                layout=cfg.V_SHARED_LAYOUT,
            )
        else:
            k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=key_cache_ptr,
                shape=(num_blocks * cfg.NUM_KV_HEADS, cfg.BLOCK_SIZE * cfg.HEAD_SIZE),
                strides=(stride_k_cache_1, 1),
                block_shape=(
                    cfg.NUM_BLOCKS_GATHER_PER_TILE,
                    cfg.BLOCK_SIZE * cfg.HEAD_SIZE,
                ),
                layout=cfg.K_SHARED_LAYOUT,
            )
            v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=value_cache_ptr,
                shape=(num_blocks * cfg.NUM_KV_HEADS, cfg.BLOCK_SIZE * cfg.HEAD_SIZE),
                strides=(stride_v_cache_1, 1),
                block_shape=(
                    cfg.NUM_BLOCKS_GATHER_PER_TILE,
                    cfg.BLOCK_SIZE * cfg.HEAD_SIZE,
                ),
                layout=cfg.V_SHARED_LAYOUT,
            )

        k_shared = gl.allocate_shared_memory(
            k_desc.dtype,
            [cfg.NUM_STAGES] + k_desc.block_shape,
            layout=cfg.K_SHARED_LAYOUT,
        )
        v_shared = gl.allocate_shared_memory(
            v_desc.dtype,
            [cfg.NUM_STAGES] + v_desc.block_shape,
            layout=cfg.V_SHARED_LAYOUT,
        )

        # Calculate tile range
        num_tiles = (max_seq_prefix_len + cfg.BLOCK_SIZE - 1) // cfg.BLOCK_SIZE
        tile_start = segm_idx * tiles_per_segment
        tile_end = min((segm_idx + 1) * tiles_per_segment, num_tiles)
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

        query_pos_qk = gl.convert_layout(
            query_pos_qk, gl.SliceLayout(1, cfg.QK_WMMA_LAYOUT)
        )[:, None]
        query_mask_qk = gl.convert_layout(query_mask_qk, cfg.QK_WMMA_LAYOUT)

        context_len_q_pos_qk = context_len + query_pos_qk

        # Compute the tile index beyond which causal masking is needed.
        # min causal pos = context_len + first query pos in block
        # Tiles j < safe_tile_end have all KV positions within causal range
        # for every query row, so apply_mask_qk can be skipped.
        min_causal_pos = context_len + q_block_local_idx * cfg.BLOCK_Q
        safe_tile_end = (min_causal_pos + 1) // cfg.BLOCK_SIZE
        safe_tile_end = gl.minimum(safe_tile_end, tile_end)
        safe_tile_end = gl.maximum(safe_tile_end, tile_start)

        return AttentionProgram(
            cfg,
            q,
            k_shared,
            v_shared,
            key_cache_ptr,
            value_cache_ptr,
            output_ptr,
            # segm_output_ptr,
            segm_max_ptr,
            segm_expsum_ptr,
            tile_start,
            tile_end,
            safe_tile_end,
            kv_head_idx,
            context_len,
            context_len_q_pos_qk,
            query_pos_qk,
            query_mask_qk,
            query_offset_0_qk,
            query_offset_1_qk,
            query_mask_0_qk,
            query_mask_1_qk,
            query_offset_0_pv,
            query_offset_1_pv,
            query_mask_0_pv,
            query_mask_1_pv,
            k_desc,
            v_desc,
            stride_k_cache_0,
            stride_k_cache_1,
            stride_k_cache_2,
            stride_k_cache_3,
            stride_v_cache_0,
            stride_v_cache_1,
            stride_v_cache_2,
            stride_v_cache_3,
            qq_bias_stride_0,
            softcap,
            q_scale,
            k_scale,
            v_scale,
        )

    @gluon.jit
    def get_next_buffer_id(self, buffer_id):
        if self.cfg.NUM_STAGES == 2:
            return 1 - buffer_id
        else:
            return (buffer_id + 1) % self.cfg.NUM_STAGES

    @gluon.jit
    def allocate_accumulator(
        self,
        sink_ptr,
        segm_idx,
        query_offset_1,
        query_mask_1,
    ):
        if self.cfg.USE_SINKS:
            if segm_idx == 0:
                # Prescale with RCP_LN2, needed for exp2
                M = (
                    gl.amd.cdna4.buffer_load(
                        ptr=sink_ptr,
                        offsets=query_offset_1.to(gl.int32),
                        mask=query_mask_1,
                        other=float("-inf"),
                    ).to(dtype=gl.float32)
                    * self.cfg.RCP_LN2
                )
            else:
                M = gl.full(
                    [self.cfg.BLOCK_M],
                    float("-inf"),
                    dtype=tl.float32,
                    layout=gl.SliceLayout(1, self.cfg.QK_WMMA_LAYOUT),
                )
        else:
            M = gl.full(
                [self.cfg.BLOCK_M],
                float("-inf"),
                dtype=tl.float32,
                layout=gl.SliceLayout(1, self.cfg.QK_WMMA_LAYOUT),
            )

        L = gl.full(
            [self.cfg.BLOCK_M],
            1.0,
            dtype=tl.float32,
            layout=gl.SliceLayout(1, self.cfg.QK_WMMA_LAYOUT),
        )
        acc = gl.zeros(
            [self.cfg.BLOCK_M, self.cfg.HEAD_SIZE],
            dtype=tl.float32,
            layout=self.cfg.PV_WMMA_LAYOUT,
        )

        return L, M, acc

    @gluon.jit
    def load_physical_block_idx(self, j, block_tables_ptr_shifted):
        if self.cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
            # TDM load
            physical_block_idx = gl.load(block_tables_ptr_shifted + j)
        else:
            # TDM gather
            offs_j = gl.arange(
                0,
                self.cfg.NUM_BLOCKS_GATHER_PER_TILE,
                layout=self.cfg.GATHER_BLOCKED_LAYOUT,
            )
            physical_block_idx = gl.load(
                block_tables_ptr_shifted
                + j * self.cfg.NUM_BLOCKS_GATHER_PER_TILE
                + offs_j
            )

        return j + 1, physical_block_idx

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
            0, self.cfg.BLOCK_M, layout=gl.SliceLayout(1, self.cfg.Q_DOT_LAYOUT)
        )
        offs_d = gl.arange(
            0, self.cfg.HEAD_SIZE, layout=gl.SliceLayout(0, self.cfg.Q_DOT_LAYOUT)
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
                cache_modifier=self.cfg.q_cache_modifier,
            )
        else:
            q = gl.load(
                query_ptr + q_offs,
                mask=query_mask,
                other=0.0,
                cache_modifier=self.cfg.q_cache_modifier,
            )
        return q, query_pos, query_mask

    @gluon.jit
    def unshuffle_k(self, K):
        K = (
            K.reshape(
                1,
                self.cfg.TILE_SIZE // 16,
                self.cfg.HEAD_SIZE // 16,
                2,
                16,
                8,
            )
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(self.cfg.TILE_SIZE, self.cfg.HEAD_SIZE)
            .trans(1, 0)
        )
        return gl.convert_layout(
            value=K, layout=self.cfg.K_DOT_LAYOUT, assert_trivial=True
        )

    @gluon.jit
    def unshuffle_v(self, V):
        V = (
            V.reshape(
                1,
                self.cfg.HEAD_SIZE // 16,
                self.cfg.TILE_SIZE // 16,
                2,
                16,
                8,
            )
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(self.cfg.HEAD_SIZE, self.cfg.TILE_SIZE)
            .trans(1, 0)
        )
        return gl.convert_layout(
            value=V, layout=self.cfg.V_DOT_LAYOUT, assert_trivial=True
        )

    @gluon.jit
    def lds_unshuffle_k(self, buffer_id):
        return (
            self.k_shared.index(buffer_id)
            .reshape(
                (
                    self.cfg.NUM_BLOCKS_GATHER_PER_TILE,
                    self.cfg.BLOCK_SIZE // 16,
                    self.cfg.HEAD_SIZE // 16,
                    2,
                    16,
                    8,
                )
            )
            .permute((0, 1, 4, 2, 3, 5))
            .reshape((self.cfg.TILE_SIZE, self.cfg.HEAD_SIZE))
            .permute((1, 0))
        )

    @gluon.jit
    def lds_unshuffle_v(self, buffer_id):
        return (
            self.v_shared.index(buffer_id)
            .reshape(
                (
                    self.cfg.NUM_BLOCKS_GATHER_PER_TILE,
                    self.cfg.HEAD_SIZE // 16,
                    self.cfg.BLOCK_SIZE // 16,
                    2,
                    16,
                    8,
                )
            )
            .permute((0, 1, 4, 2, 3, 5))
            .reshape(
                (
                    self.cfg.NUM_BLOCKS_GATHER_PER_TILE,
                    self.cfg.HEAD_SIZE,
                    self.cfg.BLOCK_SIZE,
                )
            )
            .permute((1, 0, 2))
            .reshape((self.cfg.HEAD_SIZE, self.cfg.TILE_SIZE))
            .permute((1, 0))
        )

    @gluon.jit
    def tdm_shared_load_k(self, wait_count, buffer_id):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        if self.cfg.SHUFFLED_KV_CACHE:
            return self.lds_unshuffle_k(buffer_id).load(layout=self.cfg.K_DOT_LAYOUT)
            # K = self.k_shared.index(buffer_id).load(layout=self.cfg.K_LOAD_LAYOUT)
            # return self.unshuffle_k(K)

        elif self.cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
            return (
                self.k_shared.index(buffer_id)
                .permute([1, 0])
                .load(layout=self.cfg.K_DOT_LAYOUT)
            )
        else:
            return (
                self.k_shared.index(buffer_id)
                .reshape([self.cfg.TILE_SIZE, self.cfg.HEAD_SIZE])
                .permute([1, 0])
                .load(layout=self.cfg.K_DOT_LAYOUT)
            )

    @gluon.jit
    def tdm_shared_load_v(self, wait_count, buffer_id):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        if self.cfg.SHUFFLED_KV_CACHE:
            return self.lds_unshuffle_v(buffer_id).load(layout=self.cfg.V_DOT_LAYOUT)
            # V = self.v_shared.index(buffer_id).load(layout=self.cfg.V_LOAD_LAYOUT)
            # return self.unshuffle_v(V)
        else:
            if self.cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
                return self.v_shared.index(buffer_id).load(layout=self.cfg.V_DOT_LAYOUT)
            else:
                return (
                    self.v_shared.index(buffer_id)
                    .reshape([self.cfg.TILE_SIZE, self.cfg.HEAD_SIZE])
                    .load(layout=self.cfg.V_DOT_LAYOUT)
                )

    @gluon.jit
    def tdm_load_global_to_shared_k(self, block_idx, buffer_id):
        if self.cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
            if self.cfg.SHUFFLED_KV_CACHE:
                offsets = [
                    (block_idx * self.cfg.NUM_KV_HEADS + self.kv_head_idx).to(gl.int32),
                    0,
                ]
                gl.amd.gfx1250.tdm.async_load(
                    self.k_desc, offsets, self.k_shared.index(buffer_id)
                )
            else:
                offsets = [
                    (block_idx * self.cfg.BLOCK_SIZE).to(gl.int32),
                    (self.kv_head_idx * self.stride_k_cache_2).to(gl.int32),
                ]
                gl.amd.gfx1250.tdm.async_load(
                    self.k_desc, offsets, self.k_shared.index(buffer_id)
                )
        else:
            # TDM gather handles both shuffled and unshuffled cases in the same way
            src_row_indices = (block_idx * self.cfg.NUM_KV_HEADS + self.kv_head_idx).to(
                gl.int32
            )
            gl.amd.gfx1250.tdm.async_gather(
                self.k_desc,
                src_row_indices,
                0,
                self.k_shared.index(buffer_id),
            )

    @gluon.jit
    def tdm_load_global_to_shared_v(self, block_idx, buffer_id):
        if self.cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
            if self.cfg.SHUFFLED_KV_CACHE:
                offsets = [
                    (block_idx * self.cfg.NUM_KV_HEADS + self.kv_head_idx).to(gl.int32),
                    0,
                ]
                gl.amd.gfx1250.tdm.async_load(
                    self.v_desc, offsets, self.v_shared.index(buffer_id)
                )
            else:
                offsets = [
                    (block_idx * self.cfg.BLOCK_SIZE).to(gl.int32),
                    (self.kv_head_idx * self.stride_v_cache_2).to(gl.int32),
                ]
                gl.amd.gfx1250.tdm.async_load(
                    self.v_desc, offsets, self.v_shared.index(buffer_id)
                )
        else:
            # TDM gather handles both shuffled and unshuffled cases in the same way
            src_row_indices = (block_idx * self.cfg.NUM_KV_HEADS + self.kv_head_idx).to(
                gl.int32
            )
            gl.amd.gfx1250.tdm.async_gather(
                self.v_desc,
                src_row_indices,
                0,
                self.v_shared.index(buffer_id),
            )

    @gluon.jit
    def compute_qk(self, k):
        S = gl.zeros(
            [self.cfg.BLOCK_M, self.cfg.TILE_SIZE],
            dtype=gl.float32,
            layout=self.cfg.QK_WMMA_LAYOUT,
        )
        pre_factor: gl.float32 = self.cfg.QK_SCALE
        if self.cfg.IS_Q_FP8 and self.cfg.IS_KV_FP8:
            pre_factor = pre_factor * self.q_scale * self.k_scale
        elif self.cfg.IS_KV_FP8:
            k = k.to(self.q.dtype)
            pre_factor = pre_factor * self.k_scale
        return gl.amd.gfx1250.wmma(self.q, k, S) * pre_factor

    @gluon.jit
    def apply_softcap(self, S):
        if self.cfg.USE_SOFTCAP:
            S = apply_softcap(S, self.softcap) * self.cfg.RCP_LN2
        return S

    @gluon.jit
    def apply_mask_qk_3D(self, S, seq_offset, alibi_slope, qq_bias_row_ptrs):
        seq_mask = seq_offset[None, :] < self.context_len + self.query_pos_qk + 1
        S = gl.where(self.query_mask_qk & seq_mask, S, float("-inf"))
        if self.cfg.SLIDING_WINDOW > 0:
            S = gl.where(
                (self.context_len + self.query_pos_qk - seq_offset)
                < self.cfg.SLIDING_WINDOW,
                S,
                float("-inf"),
            )

        if self.cfg.USE_ALIBI_SLOPES:
            # prescale w. RCP_LN2 for later exp2
            S += (
                alibi_slope[:, None]
                * (seq_offset - self.context_len)
                * self.cfg.RCP_LN2
            )

        if self.cfg.USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - self.context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < self.qq_bias_stride_0
            qq_bias = gl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            # prescale w. RCP_LN2 for later exp2
            S += qq_bias * self.cfg.RCP_LN2

        return S

    @gluon.jit
    def apply_mask_qk(self, S, j):
        seq_offset = (
            j * self.cfg.TILE_SIZE
            + gl.arange(
                0,
                self.cfg.TILE_SIZE,
                layout=gl.SliceLayout(0, self.cfg.QK_WMMA_LAYOUT),
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

    @gluon.jit
    def softmax_part1(self, p, L, acc, alpha):
        l_ij = gl.sum(p, 1)
        acc = acc * gl.convert_layout(alpha[:, None], layout=self.cfg.PV_WMMA_LAYOUT)
        # p = p.to(gl.bfloat16, fp_downcast_rounding="rtz")
        L = L * alpha + l_ij
        return p, L, acc

    @gluon.jit
    def compute_pv(self, p, v, acc):
        if self.cfg.IS_KV_FP8:
            p = p.to(v.dtype)
        else:
            p = p.to(gl.bfloat16, fp_downcast_rounding="rtz")
        p = gl.convert_layout(p, self.cfg.P_DOT_LAYOUT)
        acc = gl.amd.gfx1250.wmma(p, v, acc)
        if self.cfg.IS_KV_FP8:
            acc = acc * self.v_scale
        return acc

    @gluon.jit
    def store_output_3D(self, acc, M, L, segm_idx):
        # acc = gl.convert_layout(acc, layout=self.cfg.PV_WMMA_LAYOUT)
        offs_q_d = gl.arange(
            0, self.cfg.HEAD_SIZE, layout=gl.SliceLayout(0, self.cfg.PV_WMMA_LAYOUT)
        )
        dim_mask = gl.full((1,), 1, dtype=tl.int1)

        segm_output_offset = (
            self.query_offset_0_pv[:, None]
            * (
                self.cfg.NUM_QUERY_HEADS
                * self.cfg.NUM_SEGMENTS_PER_SEQ
                * self.cfg.HEAD_SIZE
            )
            + self.query_offset_1_pv[:, None]
            * (self.cfg.NUM_SEGMENTS_PER_SEQ * self.cfg.HEAD_SIZE)
            + segm_idx * self.cfg.HEAD_SIZE
            + offs_q_d[None, :]
        )
        if self.cfg.USE_STORE_BUFFER_OP:
            gl.amd.cdna4.buffer_store(
                stored_value=acc,
                ptr=self.output_ptr,
                offsets=segm_output_offset,
                mask=dim_mask[None, :]
                & self.query_mask_0_pv[:, None]
                & self.query_mask_1_pv[:, None],
            )
        else:
            gl.store(
                self.output_ptr + segm_output_offset.to(gl.int64),
                acc,
                mask=dim_mask[None, :]
                & self.query_mask_0_pv[:, None]
                & self.query_mask_1_pv[:, None],
            )

        segm_offset = (
            self.query_offset_0_qk
            * (self.cfg.NUM_QUERY_HEADS * self.cfg.NUM_SEGMENTS_PER_SEQ)
            + self.query_offset_1_qk * self.cfg.NUM_SEGMENTS_PER_SEQ
            + segm_idx
        )
        L = gl.convert_layout(L, layout=gl.SliceLayout(1, self.cfg.QK_WMMA_LAYOUT))
        M = gl.convert_layout(M, layout=gl.SliceLayout(1, self.cfg.QK_WMMA_LAYOUT))

        if self.cfg.USE_STORE_BUFFER_OP:
            gl.amd.cdna4.buffer_store(
                stored_value=M,
                ptr=self.segm_max_ptr,
                offsets=segm_offset.to(gl.int32),
                mask=self.query_mask_0_qk & self.query_mask_1_qk,
            )
            gl.amd.cdna4.buffer_store(
                stored_value=L,
                ptr=self.segm_expsum_ptr,
                offsets=segm_offset.to(gl.int32),
                mask=self.query_mask_0_qk & self.query_mask_1_qk,
            )
        else:
            gl.store(
                self.segm_max_ptr + segm_offset.to(gl.int64),
                M,
                mask=self.query_mask_0_qk & self.query_mask_1_qk,
            )
            gl.store(
                self.segm_expsum_ptr + segm_offset.to(gl.int64),
                L,
                mask=self.query_mask_0_qk & self.query_mask_1_qk,
            )

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
            0, self.cfg.BLOCK_M, layout=gl.SliceLayout(1, self.cfg.PV_WMMA_LAYOUT)
        )
        offs_d_out = gl.arange(
            0, self.cfg.HEAD_SIZE, layout=gl.SliceLayout(0, self.cfg.PV_WMMA_LAYOUT)
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
    use_q_block_mode: gl.constexpr = True,
):
    """Binary search to find the sequence index for a given query block index."""
    left = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = gl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@gluon.jit
def get_q_metadata(
    query_start_len_ptr,
    seq_idx,
    q_block_global_idx,
    BLOCK_Q: gl.constexpr,
):
    q_block_start_idx = gl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = gl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = gl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    return q_block_local_idx, cur_batch_query_len, cur_batch_in_all_start_index


@gluon.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@gluon.jit
def get_seq_metadata(
    seq_lens_ptr,
    seq_idx,
    TILE_SIZE: gl.constexpr,
    NUM_SEGMENTS_PER_SEQ: gl.constexpr,
):
    # sequence len for this particular sequence
    seq_len = gl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    return seq_len, tiles_per_segment


gluon_kernel_unified_attention_3d_tdm_repr = make_kernel_repr(
    "gluon_kernel_unified_attention_3d_tdm",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "TILE_SIZE",
        "HEAD_SIZE",
        "NUM_BLOCKS_GATHER_PER_TILE",
        "num_warps",
        "waves_per_eu",
        "num_stages",
        "ALL_DECODE",
        "SHUFFLED_KV_CACHE",
        "IS_Q_FP8",
        "IS_KV_FP8",
    ],
)


@gluon.jit(repr=gluon_kernel_unified_attention_3d_tdm_repr)
def gluon_kernel_unified_attention_3d_tdm(
    segm_output_ptr,  # [num_tokens, num_query_heads, num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, blk_size, head_size]
    value_cache_ptr,  # [num_blks, num_kv_heads, blk_size, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    q_scale_ptr,  # [1, ], float32
    k_scale_ptr,  # [1, ], float32
    v_scale_ptr,  # [1, ], float32
    softcap,  # float32
    num_seqs: gl.int32,  # int
    num_blocks: gl.int32,  # int
    query_stride_0: gl.int32,  # int
    query_stride_1: gl.int32,  # int, should be equal to head_size
    qq_bias_stride_0: gl.int32,  # int
    USE_ALIBI_SLOPES: gl.constexpr,  # bool
    USE_QQ_BIAS: gl.constexpr,  # bool
    USE_SOFTCAP: gl.constexpr,  # bool
    USE_SINKS: gl.constexpr,  # bool
    SLIDING_WINDOW: gl.constexpr,  # int
    stride_k_cache_0: gl.int32,  # int
    stride_k_cache_1: gl.int32,  # int
    stride_k_cache_2: gl.int32,  # int
    stride_k_cache_3: gl.int32,  # int
    stride_v_cache_0: gl.int32,  # int
    stride_v_cache_1: gl.int32,  # int
    stride_v_cache_2: gl.int32,  # int
    stride_v_cache_3: gl.int32,  # int
    block_table_stride: gl.int64,  # int
    query_start_len_ptr,  # [num_seqs+1]
    SCALE: gl.constexpr,  # float32
    NUM_QUERY_HEADS: gl.constexpr,  # int
    NUM_KV_HEADS: gl.constexpr,  # int
    BLOCK_SIZE: gl.constexpr,  # int
    HEAD_SIZE: gl.constexpr,  # int
    BLOCK_Q: gl.constexpr,  # int
    BLOCK_M: gl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: gl.constexpr,  # int
    WARP_SIZE: gl.constexpr,  # int
    num_warps: gl.constexpr,  # int
    waves_per_eu: gl.constexpr,  # int
    num_stages: gl.constexpr,  # int
    num_ctas: gl.constexpr = 1,  # int
    NUM_BLOCKS_GATHER_PER_TILE: gl.constexpr = 1,  # int NUM_BLOCKS_GATHER_PER_TILE > 1 for TDM gather mode
    ALL_DECODE: gl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: gl.constexpr = False,  # bool
    USE_LOAD_BUFFER_OP: gl.constexpr = False,  # bool
    USE_STORE_BUFFER_OP: gl.constexpr = False,  # bool
    IS_Q_FP8: gl.constexpr = False,  # bool
    IS_KV_FP8: gl.constexpr = False,  # bool
):
    # Build config with all layouts and derived constants
    cfg = AttentionConfig(
        HEAD_SIZE,
        BLOCK_SIZE,
        NUM_BLOCKS_GATHER_PER_TILE,
        NUM_SEGMENTS_PER_SEQ,
        BLOCK_M,
        BLOCK_Q,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        SLIDING_WINDOW,
        num_warps,
        WARP_SIZE,
        num_stages,
        SCALE,
        USE_ALIBI_SLOPES,
        USE_QQ_BIAS,
        USE_SOFTCAP,
        USE_SINKS,
        USE_LOAD_BUFFER_OP,
        USE_STORE_BUFFER_OP,
        SHUFFLED_KV_CACHE,
        IS_Q_FP8,
        IS_KV_FP8,
    )

    # Workgroup offsets
    q_block_global_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    segm_idx = gl.program_id(2)

    # Find sequence index using binary search
    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, cfg.BLOCK_Q, True
    )

    # Get query block start and local index
    q_block_local_idx, cur_batch_query_len, cur_batch_in_all_start_index = (
        get_q_metadata(
            query_start_len_ptr,
            seq_idx,
            q_block_global_idx,
            cfg.BLOCK_Q,
        )
    )

    if q_block_local_idx * cfg.BLOCK_Q >= cur_batch_query_len:
        return

    seq_len, tiles_per_segment = get_seq_metadata(
        seq_lens_ptr,
        seq_idx,
        cfg.TILE_SIZE,
        cfg.NUM_SEGMENTS_PER_SEQ,
    )

    if segm_idx * tiles_per_segment * cfg.TILE_SIZE >= seq_len:
        return

    q_scale: gl.float32 = 1.0
    k_scale: gl.float32 = 1.0
    v_scale: gl.float32 = 1.0
    if cfg.IS_Q_FP8:
        q_scale = gl.load(q_scale_ptr)
    if cfg.IS_KV_FP8:
        k_scale = gl.load(k_scale_ptr)
        v_scale = gl.load(v_scale_ptr)

    context_len = seq_len - cur_batch_query_len
    block_tables_ptr_shifted = block_tables_ptr + seq_idx * block_table_stride

    # load Q
    offs_q_m_load = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.Q_LOAD_LAYOUT))
    offs_q_d_load = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, cfg.Q_LOAD_LAYOUT))
    query_pos_load = (
        q_block_local_idx * BLOCK_Q + offs_q_m_load // cfg.NUM_QUERIES_PER_KV
    )
    query_offset_0_load = cur_batch_in_all_start_index + query_pos_load
    query_offset_1_load = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_load % cfg.NUM_QUERIES_PER_KV
    )
    query_offset_load = (
        query_offset_0_load[:, None] * query_stride_0
        + query_offset_1_load[:, None] * query_stride_1
        + offs_q_d_load[None, :]
    )
    dim_mask_load = gl.full((1,), 1, dtype=tl.int1)
    query_mask_0_load = query_pos_load < cur_batch_query_len
    query_mask_1_load = query_offset_1_load < cfg.NUM_QUERY_HEADS
    q_shared = gl.allocate_shared_memory(
        query_ptr.type.element_ty,
        shape=[BLOCK_M, HEAD_SIZE],
        layout=cfg.Q_SHARED_LAYOUT,
    )
    Q_load = gl.amd.cdna4.buffer_load(
        ptr=query_ptr,
        offsets=query_offset_load.to(gl.int32),
        mask=dim_mask_load[None, :]
        & query_mask_0_load[:, None]
        & query_mask_1_load[:, None],
        other=0.0,
    )
    q_shared.store(Q_load)
    Q = q_shared.load(layout=cfg.Q_DOT_LAYOUT)

    # define offsets and masks in QK WMMA_LAYOUT
    offs_q_m_qk = gl.arange(
        0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.QK_WMMA_LAYOUT)
    )
    query_pos_qk = (
        q_block_local_idx * cfg.BLOCK_Q + offs_q_m_qk // cfg.NUM_QUERIES_PER_KV
    )
    query_offset_0_qk = cur_batch_in_all_start_index + query_pos_qk
    query_offset_1_qk = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_qk % cfg.NUM_QUERIES_PER_KV
    )
    query_mask_0_qk = query_pos_qk < cur_batch_query_len
    query_mask_1_qk = query_offset_1_qk < cfg.NUM_QUERY_HEADS
    query_mask_qk = query_mask_1_qk[:, None] & query_mask_0_qk[:, None]

    query_offset_0_pv = gl.convert_layout(
        query_offset_0_qk, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT)
    )
    query_offset_1_pv = gl.convert_layout(
        query_offset_1_qk, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT)
    )
    query_mask_0_pv = gl.convert_layout(
        query_mask_0_qk, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT)
    )
    query_mask_1_pv = gl.convert_layout(
        query_mask_1_qk, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT)
    )

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * cfg.BLOCK_Q
        + (cfg.BLOCK_M - 1) // cfg.NUM_QUERIES_PER_KV
        + 1
    )
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)

    # TODO: resume from here
    # build program
    pgm: AttentionProgram = AttentionProgram.initialize(
        cfg,
        Q,
        key_cache_ptr,
        value_cache_ptr,
        segm_output_ptr,
        segm_max_ptr,
        segm_expsum_ptr,
        max_seq_prefix_len,
        q_block_local_idx,
        cur_batch_query_len,
        context_len,
        kv_head_idx,
        num_blocks,
        query_pos_qk,
        query_mask_qk,
        query_offset_0_qk,
        query_offset_1_qk,
        query_mask_0_qk,
        query_mask_1_qk,
        query_offset_0_pv,
        query_offset_1_pv,
        query_mask_0_pv,
        query_mask_1_pv,
        segm_idx,  # for 2D, segm_idx = 0
        tiles_per_segment,  # for 2D, tiles_per_segment = num_tiles = (max_seq_prefix_len + cfg.BLOCK_SIZE - 1) // cfg.BLOCK_SIZE
        stride_k_cache_0,
        stride_k_cache_1,
        stride_k_cache_2,
        stride_k_cache_3,
        stride_v_cache_0,
        stride_v_cache_1,
        stride_v_cache_2,
        stride_v_cache_3,
        qq_bias_stride_0,
        q_scale,
        k_scale,
        v_scale,
        softcap,
    )

    # alibi slope for this head
    alibi_slope = None
    if cfg.USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1_qk, mask=query_mask_1_qk, other=0.0
        )

    # query-query attention bias
    qq_bias_row_ptrs = None
    if cfg.USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos_qk[:, None] * qq_bias_stride_0

    L, M, acc = pgm.allocate_accumulator(
        sink_ptr,
        segm_idx,
        query_offset_1_qk,
        query_mask_1_qk,
    )

    j_from_hbm: gl.int32 = segm_idx * tiles_per_segment
    buffer_id: gl.int32 = 0
    physical_block_idx: gl.int32 = 0
    seq_offset = j_from_hbm * cfg.TILE_SIZE + gl.arange(
        0, cfg.TILE_SIZE, layout=gl.SliceLayout(0, cfg.QK_WMMA_LAYOUT)
    )

    for _ in range(cfg.NUM_STAGES - 1):
        j_from_hbm, physical_block_idx = pgm.load_physical_block_idx(
            j_from_hbm, block_tables_ptr_shifted
        )
        pgm.tdm_load_global_to_shared_k(physical_block_idx, buffer_id=buffer_id)
        pgm.tdm_load_global_to_shared_v(physical_block_idx, buffer_id=buffer_id)

    # Main attention loop over KV tiles (staged, num_stages=2)
    for j in range(pgm.tile_start, pgm.tile_end - (cfg.NUM_STAGES - 1)):
        # with gl.amd.warp_pipeline_stage("load", priority=2):
        j_from_hbm, physical_block_idx = pgm.load_physical_block_idx(
            j_from_hbm, block_tables_ptr_shifted
        )
        k = pgm.tdm_shared_load_k(
            wait_count=(cfg.NUM_STAGES - 2) * 2 + 1, buffer_id=buffer_id
        )
        next_buffer_id = pgm.get_next_buffer_id(buffer_id)
        # Prefetch next tile (shared is free since k, v are in registers)
        pgm.tdm_load_global_to_shared_k(physical_block_idx, buffer_id=next_buffer_id)
        pgm.tdm_load_global_to_shared_v(physical_block_idx, buffer_id=next_buffer_id)

        # with gl.amd.warp_pipeline_stage("qk", priority=0):
        # Compute attention for current tile
        S = pgm.compute_qk(k)
        S = pgm.apply_softcap(S)
        S = pgm.apply_mask_qk_3D(S, seq_offset, alibi_slope, qq_bias_row_ptrs)
        # if j >= pgm.safe_tile_end or SLIDING_WINDOW > 0:
        #     S = pgm.apply_mask_qk(S, j)

        # with gl.amd.warp_pipeline_stage("softmax", priority=1):
        p, alpha, M = pgm.softmax_part0(S, M)
        p, L, acc = pgm.softmax_part1(p, L, acc, alpha)

        # with gl.amd.warp_pipeline_stage("pv", priority=0):
        v = pgm.tdm_shared_load_v(
            wait_count=(cfg.NUM_STAGES - 1) * 2, buffer_id=buffer_id
        )
        acc = pgm.compute_pv(p, v, acc)

        buffer_id = next_buffer_id
        seq_offset += cfg.TILE_SIZE

    for _ in range(cfg.NUM_STAGES - 1):
        # Load k_i, v_i from shared into registers
        k = pgm.tdm_shared_load_k(
            wait_count=(cfg.NUM_STAGES - 2) * 2 + 1, buffer_id=buffer_id
        )
        # Compute attention for current tile
        S = pgm.compute_qk(k)

        S = pgm.apply_softcap(S)
        S = pgm.apply_mask_qk_3D(S, seq_offset, alibi_slope, qq_bias_row_ptrs)
        # S = pgm.apply_mask_qk(S, pgm.tile_end - 1)

        p, alpha, M = pgm.softmax_part0(S, M)
        p, L, acc = pgm.softmax_part1(p, L, acc, alpha)
        v = pgm.tdm_shared_load_v(
            wait_count=(cfg.NUM_STAGES - 2) * 2, buffer_id=buffer_id
        )
        acc = pgm.compute_pv(p, v, acc)

        seq_offset += cfg.TILE_SIZE

    # Normalize and store output, this is done in reduce kernel for 3D
    # l_recip = 1 / L[:, None]
    # acc = acc * l_recip

    pgm.store_output_3D(
        acc,
        M,
        L,
        segm_idx,
    )


@gluon.jit
def gluon_kernel_unified_attention_3d_tdm_pp(
    segm_output_ptr,  # [num_tokens, num_query_heads, num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, blk_size, head_size]
    value_cache_ptr,  # [num_blks, num_kv_heads, blk_size, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    q_scale_ptr,  # [1, ] float32
    k_scale_ptr,  # [1, ] float32
    v_scale_ptr,  # [1, ] float32
    softcap,  # float32
    num_seqs: gl.int32,  # int
    num_blocks: gl.int32,  # int
    query_stride_0: gl.int32,  # int
    query_stride_1: gl.int32,  # int, should be equal to head_size
    qq_bias_stride_0: gl.int32,  # int
    USE_ALIBI_SLOPES: gl.constexpr,  # bool
    USE_QQ_BIAS: gl.constexpr,  # bool
    USE_SOFTCAP: gl.constexpr,  # bool
    USE_SINKS: gl.constexpr,  # bool
    SLIDING_WINDOW: gl.constexpr,  # int
    stride_k_cache_0: gl.int32,  # int
    stride_k_cache_1: gl.int32,  # int
    stride_k_cache_2: gl.int32,  # int
    stride_k_cache_3: gl.int32,  # int
    stride_v_cache_0: gl.int32,  # int
    stride_v_cache_1: gl.int32,  # int
    stride_v_cache_2: gl.int32,  # int
    stride_v_cache_3: gl.int32,  # int
    block_table_stride: gl.int64,  # int
    query_start_len_ptr,  # [num_seqs+1]
    SCALE: gl.constexpr,  # float32
    NUM_QUERY_HEADS: gl.constexpr,  # int
    NUM_KV_HEADS: gl.constexpr,  # int
    BLOCK_SIZE: gl.constexpr,  # int
    HEAD_SIZE: gl.constexpr,  # int
    BLOCK_Q: gl.constexpr,  # int
    BLOCK_M: gl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: gl.constexpr,  # int
    WARP_SIZE: gl.constexpr,  # int
    num_warps: gl.constexpr,  # int
    waves_per_eu: gl.constexpr,  # int
    num_stages: gl.constexpr,  # int
    num_ctas: gl.constexpr = 1,  # int
    NUM_BLOCKS_GATHER_PER_TILE: gl.constexpr = 1,  # int NUM_BLOCKS_GATHER_PER_TILE > 1 for TDM gather mode
    ALL_DECODE: gl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: gl.constexpr = False,  # bool
    USE_LOAD_BUFFER_OP: gl.constexpr = False,  # bool
    USE_STORE_BUFFER_OP: gl.constexpr = False,  # bool
    IS_Q_FP8: gl.constexpr = False,  # bool
    IS_KV_FP8: gl.constexpr = False,  # bool
):
    # Build config with all layouts and derived constants
    cfg = AttentionConfig(
        HEAD_SIZE,
        BLOCK_SIZE,
        NUM_BLOCKS_GATHER_PER_TILE,
        NUM_SEGMENTS_PER_SEQ,
        BLOCK_M,
        BLOCK_Q,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        SLIDING_WINDOW,
        num_warps,
        WARP_SIZE,
        num_stages,
        SCALE,
        USE_ALIBI_SLOPES,
        USE_QQ_BIAS,
        USE_SOFTCAP,
        USE_SINKS,
        USE_LOAD_BUFFER_OP,
        USE_STORE_BUFFER_OP,
        SHUFFLED_KV_CACHE,
        IS_Q_FP8,
        IS_KV_FP8,
    )

    # Workgroup offsets
    q_block_global_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    segm_idx = gl.program_id(2)

    # Find sequence index using binary search
    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, cfg.BLOCK_Q, True
    )

    # Get query block start and local index
    q_block_local_idx, cur_batch_query_len, cur_batch_in_all_start_index = (
        get_q_metadata(
            query_start_len_ptr,
            seq_idx,
            q_block_global_idx,
            cfg.BLOCK_Q,
        )
    )

    if q_block_local_idx * cfg.BLOCK_Q >= cur_batch_query_len:
        return

    seq_len, tiles_per_segment = get_seq_metadata(
        seq_lens_ptr,
        seq_idx,
        cfg.TILE_SIZE,
        cfg.NUM_SEGMENTS_PER_SEQ,
    )

    if segm_idx * tiles_per_segment * cfg.TILE_SIZE >= seq_len:
        return

    q_scale: gl.float32 = 1.0
    k_scale: gl.float32 = 1.0
    v_scale: gl.float32 = 1.0
    if cfg.IS_Q_FP8:
        q_scale = gl.load(q_scale_ptr)
    if cfg.IS_KV_FP8:
        k_scale = gl.load(k_scale_ptr)
        v_scale = gl.load(v_scale_ptr)

    context_len = seq_len - cur_batch_query_len
    block_tables_ptr_shifted = block_tables_ptr + seq_idx * block_table_stride

    # load Q
    offs_q_m_load = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.Q_LOAD_LAYOUT))
    offs_q_d_load = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, cfg.Q_LOAD_LAYOUT))
    query_pos_load = (
        q_block_local_idx * BLOCK_Q + offs_q_m_load // cfg.NUM_QUERIES_PER_KV
    )
    query_offset_0_load = cur_batch_in_all_start_index + query_pos_load
    query_offset_1_load = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_load % cfg.NUM_QUERIES_PER_KV
    )
    query_offset_load = (
        query_offset_0_load[:, None] * query_stride_0
        + query_offset_1_load[:, None] * query_stride_1
        + offs_q_d_load[None, :]
    )
    dim_mask_load = gl.full((1,), 1, dtype=tl.int1)
    query_mask_0_load = query_pos_load < cur_batch_query_len
    query_mask_1_load = query_offset_1_load < cfg.NUM_QUERY_HEADS
    q_shared = gl.allocate_shared_memory(
        query_ptr.type.element_ty,
        shape=[BLOCK_M, HEAD_SIZE],
        layout=cfg.Q_SHARED_LAYOUT,
    )
    Q_load = gl.amd.cdna4.buffer_load(
        ptr=query_ptr,
        offsets=query_offset_load.to(gl.int32),
        mask=dim_mask_load[None, :]
        & query_mask_0_load[:, None]
        & query_mask_1_load[:, None],
        other=0.0,
    )
    q_shared.store(Q_load)
    Q = q_shared.load(layout=cfg.Q_DOT_LAYOUT)

    # define offsets and masks in QK WMMA_LAYOUT
    offs_q_m_qk = gl.arange(
        0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.QK_WMMA_LAYOUT)
    )
    query_pos_qk = (
        q_block_local_idx * cfg.BLOCK_Q + offs_q_m_qk // cfg.NUM_QUERIES_PER_KV
    )
    query_offset_0_qk = cur_batch_in_all_start_index + query_pos_qk
    query_offset_1_qk = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_qk % cfg.NUM_QUERIES_PER_KV
    )
    query_mask_0_qk = query_pos_qk < cur_batch_query_len
    query_mask_1_qk = query_offset_1_qk < cfg.NUM_QUERY_HEADS
    query_mask_qk = query_mask_1_qk[:, None] & query_mask_0_qk[:, None]

    query_offset_0_pv = gl.convert_layout(
        query_offset_0_qk, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT)
    )
    query_offset_1_pv = gl.convert_layout(
        query_offset_1_qk, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT)
    )
    query_mask_0_pv = gl.convert_layout(
        query_mask_0_qk, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT)
    )
    query_mask_1_pv = gl.convert_layout(
        query_mask_1_qk, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT)
    )

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * cfg.BLOCK_Q
        + (cfg.BLOCK_M - 1) // cfg.NUM_QUERIES_PER_KV
        + 1
    )
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)

    # TODO: resume from here
    # build program
    pgm: AttentionProgram = AttentionProgram.initialize(
        cfg,
        Q,
        key_cache_ptr,
        value_cache_ptr,
        segm_output_ptr,
        segm_max_ptr,
        segm_expsum_ptr,
        max_seq_prefix_len,
        q_block_local_idx,
        cur_batch_query_len,
        context_len,
        kv_head_idx,
        num_blocks,
        query_pos_qk,
        query_mask_qk,
        query_offset_0_qk,
        query_offset_1_qk,
        query_mask_0_qk,
        query_mask_1_qk,
        query_offset_0_pv,
        query_offset_1_pv,
        query_mask_0_pv,
        query_mask_1_pv,
        segm_idx,  # for 2D, segm_idx = 0
        tiles_per_segment,  # for 2D, tiles_per_segment = num_tiles = (max_seq_prefix_len + cfg.BLOCK_SIZE - 1) // cfg.BLOCK_SIZE
        stride_k_cache_0,
        stride_k_cache_1,
        stride_k_cache_2,
        stride_k_cache_3,
        stride_v_cache_0,
        stride_v_cache_1,
        stride_v_cache_2,
        stride_v_cache_3,
        qq_bias_stride_0,
        softcap,
    )

    # alibi slope for this head
    alibi_slope = None
    if cfg.USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1_qk, mask=query_mask_1_qk, other=0.0
        )

    # query-query attention bias
    qq_bias_row_ptrs = None
    if cfg.USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos_qk[:, None] * qq_bias_stride_0

    L, M, acc = pgm.allocate_accumulator(
        sink_ptr,
        segm_idx,
        query_offset_1_qk,
        query_mask_1_qk,
    )

    j_from_hbm: gl.int32 = segm_idx * tiles_per_segment
    j_from_hbm: gl.int32 = segm_idx * tiles_per_segment
    j_from_hbm: gl.int32 = segm_idx * tiles_per_segment
    buffer_id: gl.int32 = 0
    physical_block_idx: gl.int32 = 0
    seq_offset = j_from_hbm * cfg.TILE_SIZE + gl.arange(
        0, cfg.TILE_SIZE, layout=gl.SliceLayout(0, cfg.QK_WMMA_LAYOUT)
    )

    j_from_hbm, physical_block_idx = pgm.load_physical_block_idx(
        j_from_hbm, block_tables_ptr_shifted
    )
    j_from_hbm, next_physical_block_idx = pgm.load_physical_block_idx(
        j_from_hbm, block_tables_ptr_shifted
    )
    pgm.tdm_load_global_to_shared_k(physical_block_idx, buffer_id=0)
    pgm.tdm_load_global_to_shared_k(next_physical_block_idx, buffer_id=1)
    pgm.tdm_load_global_to_shared_v(physical_block_idx, buffer_id=0)

    k = pgm.tdm_shared_load_k(wait_count=2, buffer_id=0)

    S = pgm.compute_qk(k)
    S = pgm.apply_softcap(S)
    S = pgm.apply_mask_qk_3D(S, seq_offset, alibi_slope, qq_bias_row_ptrs)
    p, alpha, M = pgm.softmax_part0(S, M)

    j_from_hbm, physical_block_idx = pgm.load_physical_block_idx(
        j_from_hbm, block_tables_ptr_shifted
    )
    pgm.tdm_load_global_to_shared_v(next_physical_block_idx, buffer_id=1)
    pgm.tdm_load_global_to_shared_k(physical_block_idx, buffer_id=0)

    k = pgm.tdm_shared_load_k(wait_count=3, buffer_id=1)
    # Main attention loop over KV tiles (staged, num_stages=2)
    for j in range(pgm.tile_start, pgm.tile_end - (cfg.NUM_STAGES - 1)):

        with gl.amd.warp_pipeline_stage("qk", priority=0):
            S = pgm.compute_qk(k)

        with gl.amd.warp_pipeline_stage("softmax", priority=1):
            p, L, acc = pgm.softmax_part1(p, L, acc, alpha)
            v = pgm.tdm_shared_load_v(wait_count=2, buffer_id=buffer_id)
            p = p.to(v.dtype)
            next_buffer_id = pgm.get_next_buffer_id(buffer_id)
            j_from_hbm, next_physical_block_idx = pgm.load_physical_block_idx(
                j_from_hbm, block_tables_ptr_shifted
            )
            pgm.tdm_load_global_to_shared_k(
                next_physical_block_idx, buffer_id=next_buffer_id
            )

        with gl.amd.warp_pipeline_stage("pv", priority=0):
            acc = pgm.compute_pv(p, v, acc)

            # buffer_id = next_buffer_id
            seq_offset += cfg.TILE_SIZE

        with gl.amd.warp_pipeline_stage("pv", priority=1):
            S = pgm.apply_softcap(S)
            S = pgm.apply_mask_qk_3D(S, seq_offset, alibi_slope, qq_bias_row_ptrs)
            p, alpha, M = pgm.softmax_part0(S, M)

            k = pgm.tdm_shared_load_k(wait_count=2, buffer_id=buffer_id)

            next_buffer_id = pgm.get_next_buffer_id(buffer_id)
            pgm.tdm_load_global_to_shared_v(
                next_physical_block_idx, buffer_id=next_buffer_id
            )

    # Load k_i, v_i from shared into registers
    k = pgm.tdm_shared_load_k(
        wait_count=(cfg.NUM_STAGES - 2) * 2 + 1, buffer_id=buffer_id
    )
    # Compute attention for current tile
    S = pgm.compute_qk(k)

    S = pgm.apply_softcap(S)
    S = pgm.apply_mask_qk_3D(S, seq_offset, alibi_slope, qq_bias_row_ptrs)
    # S = pgm.apply_mask_qk(S, pgm.tile_end - 1)

    p, alpha, M = pgm.softmax_part0(S, M)
    p, L, acc = pgm.softmax_part1(p, L, acc, alpha)
    v = pgm.tdm_shared_load_v(wait_count=(cfg.NUM_STAGES - 2) * 2, buffer_id=buffer_id)
    p = p.to(v.dtype)
    acc = pgm.compute_pv(p, v, acc)

    for _ in range(cfg.NUM_STAGES - 1):
        j_from_hbm, physical_block_idx = pgm.load_physical_block_idx(
            j_from_hbm, block_tables_ptr_shifted
        )
        pgm.tdm_load_global_to_shared_k(physical_block_idx, buffer_id=buffer_id)
        pgm.tdm_load_global_to_shared_v(physical_block_idx, buffer_id=buffer_id)

    # Main attention loop over KV tiles (staged, num_stages=2)
    for j in range(pgm.tile_start, pgm.tile_end - (cfg.NUM_STAGES - 1)):
        # with gl.amd.warp_pipeline_stage("load", priority=2):
        j_from_hbm, physical_block_idx = pgm.load_physical_block_idx(
            j_from_hbm, block_tables_ptr_shifted
        )
        k = pgm.tdm_shared_load_k(
            wait_count=(cfg.NUM_STAGES - 2) * 2 + 1, buffer_id=buffer_id
        )
        next_buffer_id = pgm.get_next_buffer_id(buffer_id)
        # Prefetch next tile (shared is free since k, v are in registers)
        pgm.tdm_load_global_to_shared_k(physical_block_idx, buffer_id=next_buffer_id)
        pgm.tdm_load_global_to_shared_v(physical_block_idx, buffer_id=next_buffer_id)

        # with gl.amd.warp_pipeline_stage("qk", priority=0):
        # Compute attention for current tile
        S = pgm.compute_qk(k)
        S = pgm.apply_softcap(S)
        S = pgm.apply_mask_qk_3D(S, seq_offset, alibi_slope, qq_bias_row_ptrs)
        # if j >= pgm.safe_tile_end or SLIDING_WINDOW > 0:
        #     S = pgm.apply_mask_qk(S, j)

        # with gl.amd.warp_pipeline_stage("softmax", priority=1):
        p, alpha, M = pgm.softmax_part0(S, M)
        p, L, acc = pgm.softmax_part1(p, L, acc, alpha)

        # with gl.amd.warp_pipeline_stage("pv", priority=0):
        v = pgm.tdm_shared_load_v(
            wait_count=(cfg.NUM_STAGES - 1) * 2, buffer_id=buffer_id
        )
        p = p.to(v.dtype)
        acc = pgm.compute_pv(p, v, acc)

        buffer_id = next_buffer_id
        seq_offset += cfg.TILE_SIZE

    for _ in range(cfg.NUM_STAGES - 1):
        # Load k_i, v_i from shared into registers
        k = pgm.tdm_shared_load_k(
            wait_count=(cfg.NUM_STAGES - 2) * 2 + 1, buffer_id=buffer_id
        )
        # Compute attention for current tile
        S = pgm.compute_qk(k)

        S = pgm.apply_softcap(S)
        S = pgm.apply_mask_qk_3D(S, seq_offset, alibi_slope, qq_bias_row_ptrs)
        # S = pgm.apply_mask_qk(S, pgm.tile_end - 1)

        p, alpha, M = pgm.softmax_part0(S, M)
        p, L, acc = pgm.softmax_part1(p, L, acc, alpha)
        v = pgm.tdm_shared_load_v(
            wait_count=(cfg.NUM_STAGES - 2) * 2, buffer_id=buffer_id
        )
        p = p.to(v.dtype)
        acc = pgm.compute_pv(p, v, acc)

        seq_offset += cfg.TILE_SIZE

    # Normalize and store output, this is done in reduce kernel for 3D
    l_recip = 1 / L[:, None]
    acc = acc * l_recip

    pgm.store_output_3D(
        acc,
        M,
        L,
        segm_idx,
    )
