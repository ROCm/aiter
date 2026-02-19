# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
from re import T
import triton
import triton.language as tl
import torch
from aiter.ops.triton.utils.types import e4m3_dtype
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
import aiter.ops.triton.utils._triton.arch_info as arch_info
from triton.language.core import _aggregate as aggregate
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# from triton._C.libtriton.gluon_ir import make_cga_layout

DEVICE_ARCH = arch_info.get_arch()
MMA_operation: ttgl.constexpr = (
    ttgl.amd.gfx1250.wmma
    if ttgl.constexpr(DEVICE_ARCH in ("gfx1250",))
    else ttgl.amd.cdna4.mfma
)

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
    HEAD_SIZE: ttgl.constexpr
    BLOCK_SIZE: ttgl.constexpr
    NUM_BLOCKS_GATHER_PER_TILE: ttgl.constexpr
    NUM_SEGMENTS_PER_SEQ: ttgl.constexpr
    BLOCK_M: ttgl.constexpr
    NUM_QUERY_HEADS: ttgl.constexpr
    NUM_KV_HEADS: ttgl.constexpr
    SLIDING_WINDOW: ttgl.constexpr

    # Derived constants
    TILE_SIZE: ttgl.constexpr
    NUM_QUERIES_PER_KV: ttgl.constexpr
    BLOCK_Q: ttgl.constexpr
    RCP_LN2: ttgl.constexpr
    QK_SCALE: ttgl.constexpr

    # Operator layouts (CDNA4 MFMA)
    QK_WMMA_LAYOUT: ttgl.constexpr
    PV_WMMA_LAYOUT: ttgl.constexpr

    # Dot operand layouts
    Q_DOT_LAYOUT: ttgl.constexpr
    K_DOT_LAYOUT: ttgl.constexpr
    V_DOT_LAYOUT: ttgl.constexpr
    P_DOT_LAYOUT: ttgl.constexpr

    # Layout for loading Q
    Q_LOAD_LAYOUT: ttgl.constexpr

    # Shared memory layouts
    Q_SHARED_LAYOUT: ttgl.constexpr
    K_SHARED_LAYOUT: ttgl.constexpr
    V_SHARED_LAYOUT: ttgl.constexpr
    GATHER_BLOCKED_LAYOUT: ttgl.constexpr

    q_cache_modifier: ttgl.constexpr
    kv_cache_modifier: ttgl.constexpr

    USE_ALIBI_SLOPES: ttgl.constexpr
    USE_QQ_BIAS: ttgl.constexpr
    USE_SOFTCAP: ttgl.constexpr
    USE_SINKS: ttgl.constexpr
    USE_LOAD_BUFFER_OP: ttgl.constexpr
    USE_STORE_BUFFER_OP: ttgl.constexpr

    NUM_STAGES: ttgl.constexpr

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
    ):
        # Constants
        self.HEAD_SIZE = ttgl.constexpr(HEAD_SIZE)
        self.BLOCK_SIZE = ttgl.constexpr(BLOCK_SIZE)
        self.NUM_BLOCKS_GATHER_PER_TILE = ttgl.constexpr(NUM_BLOCKS_GATHER_PER_TILE)
        self.NUM_SEGMENTS_PER_SEQ = ttgl.constexpr(NUM_SEGMENTS_PER_SEQ)
        self.BLOCK_M = ttgl.constexpr(BLOCK_M)
        self.NUM_QUERY_HEADS = ttgl.constexpr(NUM_QUERY_HEADS)
        self.NUM_KV_HEADS = ttgl.constexpr(NUM_KV_HEADS)
        self.SLIDING_WINDOW = ttgl.constexpr(SLIDING_WINDOW)
        self.NUM_STAGES = ttgl.constexpr(NUM_STAGES)
        # Derived constants
        self.TILE_SIZE = ttgl.constexpr(BLOCK_SIZE * NUM_BLOCKS_GATHER_PER_TILE)
        self.NUM_QUERIES_PER_KV = ttgl.constexpr(NUM_QUERY_HEADS // NUM_KV_HEADS)
        self.BLOCK_Q = ttgl.constexpr(BLOCK_Q)
        self.RCP_LN2 = ttgl.constexpr(1.4426950408889634)
        self.QK_SCALE = ttgl.constexpr(SCALE * self.RCP_LN2)
        self.USE_ALIBI_SLOPES = ttgl.constexpr(USE_ALIBI_SLOPES)
        self.USE_QQ_BIAS = ttgl.constexpr(USE_QQ_BIAS)
        self.USE_SOFTCAP = ttgl.constexpr(USE_SOFTCAP)
        self.USE_SINKS = ttgl.constexpr(USE_SINKS)
        self.USE_LOAD_BUFFER_OP = ttgl.constexpr(USE_LOAD_BUFFER_OP)
        self.USE_STORE_BUFFER_OP = ttgl.constexpr(USE_STORE_BUFFER_OP)

        # ttgl.static_assert(NUM_WARPS == 2 or NUM_WARPS == 4, "NUM_WARPS must be 2 or 4")
        assert NUM_WARPS == 2 or NUM_WARPS == 4

        if NUM_WARPS == 2:
            warp_bases = [(0, 1)]
        elif NUM_WARPS == 4:
            warp_bases = [(0, 1), (0, 2)]

        # ttgl.static_assert(
        #     WARP_SIZE == 32 or WARP_SIZE == 64, "WARP_SIZE must be 32 or 64"
        # )
        assert WARP_SIZE == 32 or WARP_SIZE == 64

        if WARP_SIZE == 32:
            self.QK_WMMA_LAYOUT = ttgl.constexpr(
                ttgl.amd.AMDWMMALayout(
                    version=3,
                    transposed=True,
                    warp_bases=warp_bases,
                    reg_bases=[],
                    instr_shape=[16, 16, 32],
                )
            )

            self.PV_WMMA_LAYOUT = ttgl.constexpr(
                ttgl.amd.AMDWMMALayout(
                    version=3,
                    transposed=True,
                    warp_bases=warp_bases,
                    reg_bases=[],
                    instr_shape=[16, 16, 32],
                )
            )
            self.Q_DOT_LAYOUT = ttgl.constexpr(
                ttgl.DotOperandLayout(
                    operand_index=0, parent=self.QK_WMMA_LAYOUT, k_width=8
                )
            )
            self.K_DOT_LAYOUT = ttgl.constexpr(
                ttgl.DotOperandLayout(
                    operand_index=1, parent=self.QK_WMMA_LAYOUT, k_width=8
                )
            )
            self.P_DOT_LAYOUT = ttgl.constexpr(
                ttgl.DotOperandLayout(
                    operand_index=0, parent=self.PV_WMMA_LAYOUT, k_width=8
                )
            )
            self.V_DOT_LAYOUT = ttgl.constexpr(
                ttgl.DotOperandLayout(
                    operand_index=1, parent=self.PV_WMMA_LAYOUT, k_width=8
                )
            )
        else:
            self.QK_WMMA_LAYOUT = ttgl.constexpr(
                ttgl.amd.AMDMFMALayout(
                    version=4,
                    instr_shape=[16, 16, 32],
                    transposed=True,
                    warps_per_cta=[2, NUM_WARPS // 2],
                )
            )
            self.PV_WMMA_LAYOUT = ttgl.constexpr(
                ttgl.amd.AMDMFMALayout(
                    version=4,
                    instr_shape=[16, 16, 16],
                    transposed=True,
                    warps_per_cta=[NUM_WARPS // 2, 2],
                )
            )
            self.Q_DOT_LAYOUT = ttgl.constexpr(
                ttgl.DotOperandLayout(
                    operand_index=0, parent=self.QK_WMMA_LAYOUT, k_width=8
                )
            )
            self.K_DOT_LAYOUT = ttgl.constexpr(
                ttgl.DotOperandLayout(
                    operand_index=1, parent=self.QK_WMMA_LAYOUT, k_width=8
                )
            )
            self.P_DOT_LAYOUT = ttgl.constexpr(
                ttgl.DotOperandLayout(
                    operand_index=0, parent=self.PV_WMMA_LAYOUT, k_width=4
                )
            )
            self.V_DOT_LAYOUT = ttgl.constexpr(
                ttgl.DotOperandLayout(
                    operand_index=1, parent=self.PV_WMMA_LAYOUT, k_width=4
                )
            )

        # ttgl.static_assert(
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

        self.Q_SHARED_LAYOUT = ttgl.constexpr(
            ttgl.PaddedSharedLayout.with_identity_for(
                interval_padding_pairs=[[HEAD_SIZE, 8]],
                shape=[BLOCK_M, HEAD_SIZE],
                order=[1, 0],
            )
        )
        if NUM_BLOCKS_GATHER_PER_TILE == 1:
            self.K_SHARED_LAYOUT = ttgl.constexpr(
                ttgl.PaddedSharedLayout.with_identity_for(
                    interval_padding_pairs=[[HEAD_SIZE, 8]],
                    shape=([BLOCK_SIZE, HEAD_SIZE]),
                    order=[1, 0],
                )
            )
            self.V_SHARED_LAYOUT = ttgl.constexpr(
                ttgl.PaddedSharedLayout.with_identity_for(
                    interval_padding_pairs=[[HEAD_SIZE, 8]],
                    shape=[BLOCK_SIZE, HEAD_SIZE],
                    order=[1, 0],
                )
            )
            self.GATHER_BLOCKED_LAYOUT = ttgl.constexpr(None)
        else:
            self.K_SHARED_LAYOUT = ttgl.constexpr(
                ttgl.PaddedSharedLayout.with_identity_for(
                    interval_padding_pairs=[[BLOCK_SIZE * HEAD_SIZE, 8]],
                    shape=([NUM_BLOCKS_GATHER_PER_TILE, BLOCK_SIZE * HEAD_SIZE]),
                    order=[1, 0],
                )
            )
            self.V_SHARED_LAYOUT = ttgl.constexpr(
                ttgl.PaddedSharedLayout.with_identity_for(
                    interval_padding_pairs=[[BLOCK_SIZE * HEAD_SIZE, 8]],
                    shape=[NUM_BLOCKS_GATHER_PER_TILE, BLOCK_SIZE * HEAD_SIZE],
                    order=[1, 0],
                )
            )
            self.GATHER_BLOCKED_LAYOUT = ttgl.constexpr(
                ttgl.BlockedLayout(
                    size_per_thread=[NUM_BLOCKS_GATHER_PER_TILE],
                    threads_per_warp=[WARP_SIZE],
                    warps_per_cta=[NUM_WARPS],
                    order=[0],
                )
            )

        # size_per_thread along the fastest moving dimension is set to 8 (BF16)
        size_per_thread_fastest_dim = ttgl.constexpr(8)
        # size_per_thread * threads_per_warp along the fastest moving dimension is set to HEAD_SIZE with only 1 warp_per_cta,
        # therefore, threads_per_warp along the fastest moving dimension should be HEAD_SIZE // size_per_thread_fastest_dim
        # clamp the threads_per_warp along the fastest moving dimension to 1 ~ WARP_SIZE
        threads_per_warp_fastest_dim = max(
            min((HEAD_SIZE // size_per_thread_fastest_dim), WARP_SIZE), 1
        )

        self.Q_LOAD_LAYOUT = ttgl.constexpr(
            ttgl.BlockedLayout(
                size_per_thread=[1, size_per_thread_fastest_dim],
                threads_per_warp=[
                    WARP_SIZE // threads_per_warp_fastest_dim,
                    threads_per_warp_fastest_dim,
                ],
                warps_per_cta=[NUM_WARPS, 1],
                order=[1, 0],
            )
        )
        # self.K_BLOCKED_LAYOUT = ttgl.constexpr(
        #     ttgl.BlockedLayout(
        #         size_per_thread=[size_per_thread_fastest_dim, 1],
        #         threads_per_warp=[
        #             threads_per_warp_fastest_dim,
        #             WARP_SIZE // threads_per_warp_fastest_dim,
        #         ],
        #         warps_per_cta=[1, NUM_WARPS],
        #         order=[0, 1],
        #     )
        # )

        self.q_cache_modifier = ttgl.constexpr(".cg")
        self.kv_cache_modifier = ttgl.constexpr("")


@aggregate
class AttentionProgram:
    """Program state and core operations for the unified attention kernel."""

    cfg: AttentionConfig

    q: ttgl.tensor
    k_shared: ttgl.shared_memory_descriptor
    v_shared: ttgl.shared_memory_descriptor

    key_cache_ptr: ttgl.tensor
    value_cache_ptr: ttgl.tensor
    output_ptr: ttgl.tensor
    # segm_output_ptr: ttgl.tensor
    segm_max_ptr: ttgl.tensor
    segm_expsum_ptr: ttgl.tensor

    tile_start: ttgl.tensor
    tile_end: ttgl.tensor
    safe_tile_end: ttgl.tensor
    kv_head_idx: ttgl.tensor
    query_mask_qk: ttgl.tensor
    context_len: ttgl.tensor
    context_len_q_pos_qk: ttgl.tensor
    query_pos_qk: ttgl.tensor
    query_mask_qk: ttgl.tensor
    query_offset_0_qk: ttgl.tensor
    query_offset_1_qk: ttgl.tensor
    query_mask_0_qk: ttgl.tensor
    query_mask_1_qk: ttgl.tensor

    k_desc: ttgl.amd.gfx1250.tdm.tensor_descriptor
    v_desc: ttgl.amd.gfx1250.tdm.tensor_descriptor
    stride_k_cache_0: ttgl.tensor
    stride_k_cache_1: ttgl.tensor
    stride_k_cache_2: ttgl.tensor
    stride_k_cache_3: ttgl.tensor
    stride_v_cache_0: ttgl.tensor
    stride_v_cache_1: ttgl.tensor
    stride_v_cache_2: ttgl.tensor
    stride_v_cache_3: ttgl.tensor

    qq_bias_stride_0: ttgl.tensor
    softcap: ttgl.tensor

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
        self.softcap = softcap

    @gluon.jit
    def initialize(
        cfg,
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
        softcap,
    ):
        # the last dimension of the stride should always be 1
        # ttgl.static_assert(stride_k_cache_3 == 1)
        # ttgl.static_assert(stride_v_cache_3 == 1)
        # if cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
        #     # in TDM mode, KV cache shape should be [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE]
        #     ttgl.static_assert(stride_k_cache_0 // stride_k_cache_1 == cfg.BLOCK_SIZE)
        #     ttgl.static_assert(stride_v_cache_0 // stride_v_cache_1 == cfg.BLOCK_SIZE)
        #     ttgl.static_assert(stride_k_cache_2 == cfg.HEAD_SIZE)
        #     ttgl.static_assert(stride_v_cache_2 == cfg.HEAD_SIZE)
        # else:
        #     # in TDM gather mode, KV cache shape should be [num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE]
        #     ttgl.static_assert(stride_k_cache_0 // stride_k_cache_1 == cfg.NUM_KV_HEADS)
        #     ttgl.static_assert(stride_v_cache_0 // stride_v_cache_1 == cfg.NUM_KV_HEADS)

        if cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
            k_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=key_cache_ptr,
                shape=(num_blocks * cfg.BLOCK_SIZE, cfg.NUM_KV_HEADS * cfg.HEAD_SIZE),
                strides=(stride_k_cache_1, stride_k_cache_3),
                block_shape=(cfg.BLOCK_SIZE, cfg.HEAD_SIZE),
                layout=cfg.K_SHARED_LAYOUT,
            )
            v_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=value_cache_ptr,
                shape=(num_blocks * cfg.BLOCK_SIZE, cfg.NUM_KV_HEADS * cfg.HEAD_SIZE),
                strides=(stride_v_cache_1, stride_v_cache_3),
                block_shape=(cfg.BLOCK_SIZE, cfg.HEAD_SIZE),
                layout=cfg.V_SHARED_LAYOUT,
            )
        else:
            k_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=key_cache_ptr,
                shape=(num_blocks * cfg.NUM_KV_HEADS, cfg.BLOCK_SIZE * cfg.HEAD_SIZE),
                strides=(stride_k_cache_1, stride_k_cache_3),
                block_shape=(
                    cfg.NUM_BLOCKS_GATHER_PER_TILE,
                    cfg.BLOCK_SIZE * cfg.HEAD_SIZE,
                ),
                layout=cfg.K_SHARED_LAYOUT,
            )
            v_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=value_cache_ptr,
                shape=(num_blocks * cfg.NUM_KV_HEADS, cfg.BLOCK_SIZE * cfg.HEAD_SIZE),
                strides=(stride_k_cache_1, stride_k_cache_3),
                block_shape=(
                    cfg.NUM_BLOCKS_GATHER_PER_TILE,
                    cfg.BLOCK_SIZE * cfg.HEAD_SIZE,
                ),
                layout=cfg.V_SHARED_LAYOUT,
            )

        k_shared = ttgl.allocate_shared_memory(
            k_desc.dtype,
            [cfg.NUM_STAGES] + k_desc.block_shape,
            layout=cfg.K_SHARED_LAYOUT,
        )
        v_shared = ttgl.allocate_shared_memory(
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
            qpos_hi = ttgl.minimum(
                qpos_lo + (cfg.BLOCK_M - 1) // cfg.NUM_QUERIES_PER_KV,
                cur_batch_query_len - 1,
            )
            first_allowed_key = context_len + qpos_lo - cfg.SLIDING_WINDOW + 1
            last_allowed_key = context_len + qpos_hi
            tile_start = ttgl.maximum(0, first_allowed_key // cfg.BLOCK_SIZE)
            tile_end = ttgl.minimum((last_allowed_key // cfg.BLOCK_SIZE) + 1, num_tiles)

        query_pos_qk = ttgl.convert_layout(
            query_pos_qk, ttgl.SliceLayout(1, cfg.QK_WMMA_LAYOUT)
        )[:, None]
        query_mask_qk = ttgl.convert_layout(query_mask_qk, cfg.QK_WMMA_LAYOUT)

        context_len_q_pos_qk = context_len + query_pos_qk

        # Compute the tile index beyond which causal masking is needed.
        # min causal pos = context_len + first query pos in block
        # Tiles j < safe_tile_end have all KV positions within causal range
        # for every query row, so apply_mask_qk can be skipped.
        min_causal_pos = context_len + q_block_local_idx * cfg.BLOCK_Q
        safe_tile_end = (min_causal_pos + 1) // cfg.BLOCK_SIZE
        safe_tile_end = ttgl.minimum(safe_tile_end, tile_end)
        safe_tile_end = ttgl.maximum(safe_tile_end, tile_start)

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
        query_pos,
        alibi_slopes_ptr,
        qq_bias_ptr,
        qq_bias_stride_0,
    ):
        if self.cfg.USE_SINKS:
            if segm_idx == 0:
                # Prescale with RCP_LN2, needed for exp2
                M = (
                    ttgl.amd.cdna4.buffer_load(
                        ptr=sink_ptr,
                        offsets=query_offset_1.to(ttgl.int32),
                        mask=query_mask_1,
                        other=float("-inf"),
                    ).to(dtype=ttgl.float32)
                    * self.cfg.RCP_LN2
                )
            else:
                M = ttgl.full(
                    [self.cfg.BLOCK_M],
                    float("-inf"),
                    dtype=tl.float32,
                    layout=ttgl.SliceLayout(1, self.cfg.PV_WMMA_LAYOUT),
                )
        else:
            M = ttgl.full(
                [self.cfg.BLOCK_M],
                float("-inf"),
                dtype=tl.float32,
                layout=ttgl.SliceLayout(1, self.cfg.PV_WMMA_LAYOUT),
            )

        L = ttgl.full(
            [self.cfg.BLOCK_M],
            1.0,
            dtype=tl.float32,
            layout=ttgl.SliceLayout(1, self.cfg.PV_WMMA_LAYOUT),
        )
        acc = ttgl.zeros(
            [self.cfg.BLOCK_M, self.cfg.HEAD_SIZE],
            dtype=tl.float32,
            layout=self.cfg.PV_WMMA_LAYOUT,
        )

        return L, M, acc

    @gluon.jit
    def load_physical_block_idx(self, j, offs_j, block_tables_ptr_shifted):
        if self.cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
            # TDM load
            physical_block_idx = ttgl.load(block_tables_ptr_shifted + j)
        else:
            # TDM gather
            offs_j = ttgl.arange(
                0,
                self.cfg.NUM_BLOCKS_GATHER_PER_TILE,
                layout=self.cfg.GATHER_BLOCKED_LAYOUT,
            )
            physical_block_idx = ttgl.load(
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
        offs_m = ttgl.arange(
            0, self.cfg.BLOCK_M, layout=ttgl.SliceLayout(1, self.cfg.Q_DOT_LAYOUT)
        )
        offs_d = ttgl.arange(
            0, self.cfg.HEAD_SIZE, layout=ttgl.SliceLayout(0, self.cfg.Q_DOT_LAYOUT)
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
            q = ttgl.amd.cdna4.buffer_load(
                query_ptr + q_offs,
                mask=query_mask,
                other=0.0,
                cache_modifier=self.cfg.q_cache_modifier,
            )
        else:
            q = ttgl.load(
                query_ptr + q_offs,
                mask=query_mask,
                other=0.0,
                cache_modifier=self.cfg.q_cache_modifier,
            )
        return q, query_pos, query_mask

    @gluon.jit
    def tdm_shared_load_k(self, wait_count, buffer_id):
        ttgl.amd.gfx1250.tdm.async_wait(wait_count)
        if self.cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
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
        ttgl.amd.gfx1250.tdm.async_wait(wait_count)
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
            offsets = [
                (block_idx * self.cfg.BLOCK_SIZE).to(ttgl.int32),
                (self.kv_head_idx * self.stride_k_cache_2).to(ttgl.int32),
            ]
            ttgl.amd.gfx1250.tdm.async_load(
                self.k_desc, offsets, self.k_shared.index(buffer_id)
            )
        else:
            src_row_indices = (block_idx * self.cfg.NUM_KV_HEADS + self.kv_head_idx).to(
                ttgl.int32
            )
            ttgl.amd.gfx1250.tdm.async_gather(
                self.k_desc,
                src_row_indices,
                0,
                self.k_shared.index(buffer_id),
            )

    @gluon.jit
    def tdm_load_global_to_shared_v(self, block_idx, buffer_id):
        if self.cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
            offsets = [
                (block_idx * self.cfg.BLOCK_SIZE).to(ttgl.int32),
                (self.kv_head_idx * self.stride_v_cache_2).to(ttgl.int32),
            ]
            ttgl.amd.gfx1250.tdm.async_load(
                self.v_desc, offsets, self.v_shared.index(buffer_id)
            )
        else:
            src_row_indices = (block_idx * self.cfg.NUM_KV_HEADS + self.kv_head_idx).to(
                ttgl.int32
            )
            ttgl.amd.gfx1250.tdm.async_gather(
                self.v_desc,
                src_row_indices,
                0,
                self.v_shared.index(buffer_id),
            )

    @gluon.jit
    def compute_qk(self, k):
        S = ttgl.zeros(
            [self.cfg.BLOCK_M, self.cfg.TILE_SIZE],
            dtype=ttgl.float32,
            layout=self.cfg.QK_WMMA_LAYOUT,
        )
        return ttgl.amd.gfx1250.wmma(self.q, k, S) * self.cfg.QK_SCALE

    @gluon.jit
    def apply_softcap(self, S):
        if self.cfg.USE_SOFTCAP:
            S = apply_softcap(S, self.softcap) * self.cfg.RCP_LN2
        return S

    @gluon.jit
    def apply_mask_qk_3D(self, S, seq_offset, alibi_slope, qq_bias_row_ptrs):
        seq_mask = seq_offset[None, :] < self.context_len + self.query_pos_qk + 1
        S = ttgl.where(self.query_mask_qk & seq_mask, S, float("-inf"))
        if self.cfg.SLIDING_WINDOW > 0:
            S = ttgl.where(
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
            qq_bias = ttgl.load(
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
            + ttgl.arange(
                0,
                self.cfg.TILE_SIZE,
                layout=ttgl.SliceLayout(0, self.cfg.QK_WMMA_LAYOUT),
            )[None, :]
        )

        seq_mask = seq_offset <= self.context_len_q_pos_qk
        if self.cfg.SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (self.context_len_q_pos_qk - seq_offset) < self.cfg.SLIDING_WINDOW
            )
        full_mask = seq_mask
        S = ttgl.where(full_mask, S, float("-inf"))
        return S

    @gluon.jit
    def softmax_part0(self, S, M):
        m_ij = ttgl.maximum(M, ttgl.max(S, axis=1))
        m_ij = ttgl.where(m_ij > float("-inf"), m_ij, 0.0)
        p = ttgl.exp2(S - m_ij[:, None])
        alpha = ttgl.exp2(M - m_ij)
        return p, alpha, m_ij

    @gluon.jit
    def softmax_part1(self, p, L, acc, alpha):
        l_ij = ttgl.sum(p, 1)
        acc = acc * alpha[:, None]
        p = p.to(ttgl.bfloat16, fp_downcast_rounding="rtz")
        L = L * alpha + l_ij
        return p, L, acc

    @gluon.jit
    def compute_pv(self, p, v, acc):
        p = ttgl.convert_layout(p, self.cfg.P_DOT_LAYOUT)
        return ttgl.amd.gfx1250.wmma(p, v, acc)

    @gluon.jit
    def store_output_3D(self, acc, M, L, segm_idx):
        # acc = ttgl.convert_layout(acc, layout=self.cfg.PV_WMMA_LAYOUT)
        offs_q_d = ttgl.arange(
            0, self.cfg.HEAD_SIZE, layout=ttgl.SliceLayout(0, self.cfg.PV_WMMA_LAYOUT)
        )
        dim_mask = ttgl.full((1,), 1, dtype=tl.int1)

        segm_output_offset = (
            self.query_offset_0_qk[:, None]
            * (
                self.cfg.NUM_QUERY_HEADS
                * self.cfg.NUM_SEGMENTS_PER_SEQ
                * self.cfg.HEAD_SIZE
            )
            + self.query_offset_1_qk[:, None]
            * (self.cfg.NUM_SEGMENTS_PER_SEQ * self.cfg.HEAD_SIZE)
            + segm_idx * self.cfg.HEAD_SIZE
            + offs_q_d[None, :]
        )
        if self.cfg.USE_STORE_BUFFER_OP:
            ttgl.amd.cdna4.buffer_store(
                stored_value=acc,
                ptr=self.output_ptr,
                offsets=segm_output_offset,
                mask=dim_mask[None, :]
                & self.query_mask_0_qk[:, None]
                & self.query_mask_1_qk[:, None],
            )
        else:
            ttgl.store(
                self.output_ptr + segm_output_offset.to(ttgl.int64),
                acc,
                mask=dim_mask[None, :]
                & self.query_mask_0_qk[:, None]
                & self.query_mask_1_qk[:, None],
            )

        segm_offset = (
            self.query_offset_0_qk
            * (self.cfg.NUM_QUERY_HEADS * self.cfg.NUM_SEGMENTS_PER_SEQ)
            + self.query_offset_1_qk * self.cfg.NUM_SEGMENTS_PER_SEQ
            + segm_idx
        )
        L = ttgl.convert_layout(L, layout=ttgl.SliceLayout(1, self.cfg.PV_WMMA_LAYOUT))
        M = ttgl.convert_layout(M, layout=ttgl.SliceLayout(1, self.cfg.PV_WMMA_LAYOUT))

        if self.cfg.USE_STORE_BUFFER_OP:
            ttgl.amd.cdna4.buffer_store(
                stored_value=M,
                ptr=self.segm_max_ptr,
                offsets=segm_offset.to(ttgl.int32),
                mask=self.query_mask_0_qk & self.query_mask_1_qk,
            )
            ttgl.amd.cdna4.buffer_store(
                stored_value=L,
                ptr=self.segm_expsum_ptr,
                offsets=segm_offset.to(ttgl.int32),
                mask=self.query_mask_0_qk & self.query_mask_1_qk,
            )
        else:
            ttgl.store(
                self.segm_max_ptr + segm_offset.to(ttgl.int64),
                M,
                mask=self.query_mask_0_qk & self.query_mask_1_qk,
            )
            ttgl.store(
                self.segm_expsum_ptr + segm_offset.to(ttgl.int64),
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
        offs_m_out = ttgl.arange(
            0, self.cfg.BLOCK_M, layout=ttgl.SliceLayout(1, self.cfg.PV_WMMA_LAYOUT)
        )
        offs_d_out = ttgl.arange(
            0, self.cfg.HEAD_SIZE, layout=ttgl.SliceLayout(0, self.cfg.PV_WMMA_LAYOUT)
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
            ttgl.amd.cdna4.buffer_store(
                casted_out, self.output_ptr, o_offs, mask=o_mask
            )
        else:
            ttgl.store(self.output_ptr + o_offs, casted_out, mask=o_mask)


@gluon.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: ttgl.constexpr,
    use_q_block_mode: ttgl.constexpr = True,
):
    """Binary search to find the sequence index for a given query block index."""
    left = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = ttgl.load(query_start_len_ptr + mid)
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
    BLOCK_Q: ttgl.constexpr,
):
    q_block_start_idx = ttgl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = ttgl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = ttgl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    return q_block_local_idx, cur_batch_query_len, cur_batch_in_all_start_index


@gluon.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@gluon.jit
def get_seq_metadata(
    seq_lens_ptr,
    seq_idx,
    TILE_SIZE: ttgl.constexpr,
    NUM_SEGMENTS_PER_SEQ: ttgl.constexpr,
):
    # sequence len for this particular sequence
    seq_len = ttgl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    return seq_len, tiles_per_segment


@gluon.jit
def kernel_unified_attention_3d_new(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size]
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
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_seqs,  # int
    num_blocks: ttgl.int32,  # int
    query_stride_0: ttgl.int32,  # int
    query_stride_1: ttgl.int32,  # int, should be equal to head_size
    qq_bias_stride_0: ttgl.int32,  # int
    USE_ALIBI_SLOPES: ttgl.constexpr,  # bool
    USE_QQ_BIAS: ttgl.constexpr,  # bool
    USE_SOFTCAP: ttgl.constexpr,  # bool
    USE_SINKS: ttgl.constexpr,  # bool
    SLIDING_WINDOW: ttgl.constexpr,  # int
    stride_k_cache_0: ttgl.int32,  # int
    stride_k_cache_1: ttgl.int32,  # int
    stride_k_cache_2: ttgl.int32,  # int
    stride_k_cache_3: ttgl.int32,  # int
    stride_v_cache_0: ttgl.int32,  # int
    stride_v_cache_1: ttgl.int32,  # int
    stride_v_cache_2: ttgl.int32,  # int
    stride_v_cache_3: ttgl.int32,  # int
    block_table_stride: ttgl.int64,  # int
    query_start_len_ptr,  # [num_seqs+1]
    SCALE: ttgl.constexpr,  # float32
    NUM_QUERY_HEADS: ttgl.constexpr,  # int
    NUM_KV_HEADS: ttgl.constexpr,  # int
    BLOCK_SIZE: ttgl.constexpr,  # int
    HEAD_SIZE: ttgl.constexpr,  # int
    BLOCK_Q: ttgl.constexpr,  # int
    BLOCK_M: ttgl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: ttgl.constexpr,  # int
    WARP_SIZE: ttgl.constexpr,  # int
    num_warps: ttgl.constexpr,  # int
    waves_per_eu: ttgl.constexpr,  # int
    NUM_STAGES: ttgl.constexpr,  # int
    num_ctas: ttgl.constexpr = 1,  # int
    NUM_BLOCKS_GATHER_PER_TILE: ttgl.constexpr = 1,  # int NUM_BLOCKS_GATHER_PER_TILE > 1 for TDM gather mode
    ALL_DECODE: ttgl.constexpr = False,  # bool
    USE_LOAD_BUFFER_OP: ttgl.constexpr = False,  # bool
    USE_STORE_BUFFER_OP: ttgl.constexpr = False,  # bool
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
        NUM_STAGES,
        SCALE,
        USE_ALIBI_SLOPES,
        USE_QQ_BIAS,
        USE_SOFTCAP,
        USE_SINKS,
        USE_LOAD_BUFFER_OP,
        USE_STORE_BUFFER_OP,
    )

    # Workgroup offsets
    q_block_global_idx = ttgl.program_id(0)
    kv_head_idx = ttgl.program_id(1)
    segm_idx = ttgl.program_id(2)

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

    context_len = seq_len - cur_batch_query_len
    block_tables_ptr_shifted = block_tables_ptr + seq_idx * block_table_stride

    # load Q
    offs_q_m_load = ttgl.arange(
        0, BLOCK_M, layout=ttgl.SliceLayout(1, cfg.Q_LOAD_LAYOUT)
    )
    offs_q_d_load = ttgl.arange(
        0, HEAD_SIZE, layout=ttgl.SliceLayout(0, cfg.Q_LOAD_LAYOUT)
    )
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
    dim_mask_load = ttgl.full((1,), 1, dtype=tl.int1)
    query_mask_0_load = query_pos_load < cur_batch_query_len
    query_mask_1_load = query_offset_1_load < cfg.NUM_QUERY_HEADS
    q_shared = ttgl.allocate_shared_memory(
        query_ptr.type.element_ty,
        shape=[BLOCK_M, HEAD_SIZE],
        layout=cfg.Q_SHARED_LAYOUT,
    )
    Q_load = ttgl.amd.cdna4.buffer_load(
        ptr=query_ptr,
        offsets=query_offset_load.to(ttgl.int32),
        mask=dim_mask_load[None, :]
        & query_mask_0_load[:, None]
        & query_mask_1_load[:, None],
        other=0.0,
    )
    q_shared.store(Q_load)
    Q = q_shared.load(layout=cfg.Q_DOT_LAYOUT)

    # define offsets and masks in QK WMMA_LAYOUT
    offs_q_m_qk = ttgl.arange(
        0, cfg.BLOCK_M, layout=ttgl.SliceLayout(1, cfg.QK_WMMA_LAYOUT)
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

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * cfg.BLOCK_Q
        + (cfg.BLOCK_M - 1) // cfg.NUM_QUERIES_PER_KV
        + 1
    )
    max_seq_prefix_len = ttgl.minimum(max_seq_prefix_len, seq_len)

    # TODO: resume from here
    # build program
    pgm = AttentionProgram.initialize(
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
        query_pos_qk,
        alibi_slopes_ptr,
        qq_bias_ptr,
        qq_bias_stride_0,
    )

    j_producer: ttgl.int32 = segm_idx * tiles_per_segment
    buffer_id: ttgl.int32 = 0
    seq_offset = j_producer * cfg.TILE_SIZE + ttgl.arange(
        0, cfg.TILE_SIZE, layout=ttgl.SliceLayout(0, cfg.QK_WMMA_LAYOUT)
    )

    for _ in range(cfg.NUM_STAGES - 1):
        j_producer, physical_block_idx = pgm.load_physical_block_idx(
            j_producer, kv_head_idx, block_tables_ptr_shifted
        )
        pgm.tdm_load_global_to_shared_k(physical_block_idx, buffer_id=buffer_id)
        pgm.tdm_load_global_to_shared_v(physical_block_idx, buffer_id=buffer_id)

    # Main attention loop over KV tiles (staged, num_stages=2)
    for j in range(pgm.tile_start, pgm.tile_end - (cfg.NUM_STAGES - 1)):
        j_producer, next_physical_block_idx = pgm.load_physical_block_idx(
            j_producer, kv_head_idx, block_tables_ptr_shifted
        )
        k = pgm.tdm_shared_load_k(
            wait_count=(cfg.NUM_STAGES - 2) * 2 + 1, buffer_id=buffer_id
        )

        next_buffer_id = pgm.get_next_buffer_id(buffer_id)
        # Prefetch next tile (shared is free since k, v are in registers)
        pgm.tdm_load_global_to_shared_k(
            next_physical_block_idx, buffer_id=next_buffer_id
        )
        pgm.tdm_load_global_to_shared_v(
            next_physical_block_idx, buffer_id=next_buffer_id
        )

        # Compute attention for current tile
        S = pgm.compute_qk(k)

        S = pgm.apply_softcap(S)
        S = pgm.apply_mask_qk_3D(S, seq_offset, alibi_slope, qq_bias_row_ptrs)
        # if j >= pgm.safe_tile_end or SLIDING_WINDOW > 0:
        #     S = pgm.apply_mask_qk(S, j)

        p, alpha, M = pgm.softmax_part0(S, M)
        p, L, acc = pgm.softmax_part1(p, L, acc, alpha)
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
