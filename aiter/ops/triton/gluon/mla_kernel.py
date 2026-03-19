# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import triton
import triton.language as tl
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from aiter.ops.triton.utils.types import e4m3_dtype
import aiter.ops.triton.utils._triton.arch_info as arch_info
from triton.language.core import _aggregate as aggregate
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

float8_info = torch.finfo(e4m3_dtype)


@aggregate
class MLAConfig:
    """Configuration for unified attention layouts and derived constants."""

    # Core dimensions
    BLOCK_SIZE: gl.constexpr
    KV_LORA_RANK: gl.constexpr
    QK_ROPE_HEAD_DIM: gl.constexpr
    NUM_BLOCKS_GATHER_PER_TILE: gl.constexpr
    NUM_SEGMENTS_PER_SEQ: gl.constexpr
    BLOCK_M: gl.constexpr
    NUM_QUERY_HEADS: gl.constexpr
    NUM_KV_HEADS: gl.constexpr

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
    Q_LORA_LOAD_LAYOUT: gl.constexpr
    Q_ROPE_LOAD_LAYOUT: gl.constexpr

    # Shared memory layouts
    Q_LORA_SHARED_LAYOUT: gl.constexpr
    Q_ROPE_SHARED_LAYOUT: gl.constexpr
    KV_LORA_SHARED_LAYOUT: gl.constexpr
    K_ROPE_SHARED_LAYOUT: gl.constexpr
    GATHER_BLOCKED_LAYOUT: gl.constexpr

    q_cache_modifier: gl.constexpr
    kv_cache_modifier: gl.constexpr

    USE_LOAD_BUFFER_OP: gl.constexpr
    USE_STORE_BUFFER_OP: gl.constexpr

    NUM_STAGES: gl.constexpr
    SHUFFLED_KV_CACHE: gl.constexpr
    IS_Q_FP8: gl.constexpr
    IS_KV_FP8: gl.constexpr
    K_WIDTH: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        BLOCK_SIZE,
        NUM_BLOCKS_GATHER_PER_TILE,
        NUM_SEGMENTS_PER_SEQ,
        BLOCK_M,
        BLOCK_Q,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        NUM_WARPS,
        WARP_SIZE,
        NUM_STAGES,
        SCALE,
        USE_LOAD_BUFFER_OP,
        USE_STORE_BUFFER_OP,
        SHUFFLED_KV_CACHE,
        IS_Q_FP8,
        IS_KV_FP8,
    ):
        # Constants
        self.KV_LORA_RANK = gl.constexpr(KV_LORA_RANK)
        self.QK_ROPE_HEAD_DIM = gl.constexpr(QK_ROPE_HEAD_DIM)
        self.BLOCK_SIZE = gl.constexpr(BLOCK_SIZE)
        self.NUM_BLOCKS_GATHER_PER_TILE = gl.constexpr(NUM_BLOCKS_GATHER_PER_TILE)
        self.NUM_SEGMENTS_PER_SEQ = gl.constexpr(NUM_SEGMENTS_PER_SEQ)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.NUM_QUERY_HEADS = gl.constexpr(NUM_QUERY_HEADS)
        self.NUM_KV_HEADS = gl.constexpr(NUM_KV_HEADS)
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
        self.USE_LOAD_BUFFER_OP = gl.constexpr(USE_LOAD_BUFFER_OP)
        self.USE_STORE_BUFFER_OP = gl.constexpr(USE_STORE_BUFFER_OP)

        assert WARP_SIZE == 32

        assert NUM_WARPS == 1 or NUM_WARPS == 2 or NUM_WARPS == 4 or NUM_WARPS == 8

        if NUM_WARPS == 1:
            warp_bases_qk = []
            warp_bases_pv = []
        elif NUM_WARPS == 2:
            warp_bases_qk = [(1, 0)]
            warp_bases_pv = [(0, 1)]
        elif NUM_WARPS == 4:
            warp_bases_qk = [(1, 0), (2, 0)]
            warp_bases_pv = [(0, 1), (0, 2)]
        elif NUM_WARPS == 8:
            warp_bases_qk = [(1, 0), (2, 0), (4, 0)]
            warp_bases_pv = [(0, 1), (0, 2), (0, 4)]

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
        k_width = 16 if self.IS_KV_FP8 and self.SHUFFLED_KV_CACHE else 8
        self.K_WIDTH = gl.constexpr(k_width)
        self.Q_DOT_LAYOUT = gl.constexpr(
            gl.DotOperandLayout(
                operand_index=0, parent=self.QK_WMMA_LAYOUT, k_width=k_width
            )
        )
        self.K_DOT_LAYOUT = gl.constexpr(
            gl.DotOperandLayout(
                operand_index=1, parent=self.QK_WMMA_LAYOUT, k_width=k_width
            )
        )
        self.P_DOT_LAYOUT = gl.constexpr(
            gl.DotOperandLayout(
                operand_index=0, parent=self.PV_WMMA_LAYOUT, k_width=k_width
            )
        )
        self.V_DOT_LAYOUT = gl.constexpr(
            gl.DotOperandLayout(
                operand_index=1, parent=self.PV_WMMA_LAYOUT, k_width=k_width
            )
        )

        assert NUM_BLOCKS_GATHER_PER_TILE == 1

        self.Q_LORA_SHARED_LAYOUT = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for(
                interval_padding_pairs=[[KV_LORA_RANK, 8]],
                shape=[BLOCK_M, KV_LORA_RANK],
                order=[1, 0],
            )
        )
        self.Q_ROPE_SHARED_LAYOUT = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for(
                interval_padding_pairs=[[QK_ROPE_HEAD_DIM, 8]],
                shape=[BLOCK_M, QK_ROPE_HEAD_DIM],
                order=[1, 0],
            )
        )

        self.GATHER_BLOCKED_LAYOUT = gl.constexpr(None)
        if self.SHUFFLED_KV_CACHE:
            self.KV_LORA_SHARED_LAYOUT = gl.constexpr(
                gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
            )
            self.K_ROPE_SHARED_LAYOUT = gl.constexpr(
                gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
            )
        else:
            self.KV_LORA_SHARED_LAYOUT = gl.constexpr(
                gl.PaddedSharedLayout.with_identity_for(
                    interval_padding_pairs=[[KV_LORA_RANK, 8]],
                    shape=([BLOCK_SIZE, KV_LORA_RANK]),
                    order=[1, 0],
                )
            )
            self.K_ROPE_SHARED_LAYOUT = gl.constexpr(
                gl.PaddedSharedLayout.with_identity_for(
                    interval_padding_pairs=[[QK_ROPE_HEAD_DIM, 8]],
                    shape=[BLOCK_SIZE, QK_ROPE_HEAD_DIM],
                    order=[1, 0],
                )
            )

        # size_per_thread along the fastest moving dimension is set to 8 (BF16)
        size_per_thread_fastest_dim = gl.constexpr(8)
        # size_per_thread * threads_per_warp along the fastest moving dimension is set to HEAD_SIZE with only 1 warp_per_cta,
        # therefore, threads_per_warp along the fastest moving dimension should be HEAD_SIZE // size_per_thread_fastest_dim
        # clamp the threads_per_warp along the fastest moving dimension to 1 ~ WARP_SIZE
        threads_per_warp_fastest_dim = max(
            min((KV_LORA_RANK // size_per_thread_fastest_dim), WARP_SIZE), 1
        )
        self.Q_LORA_LOAD_LAYOUT = gl.constexpr(
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
        threads_per_warp_fastest_dim1 = max(
            min((QK_ROPE_HEAD_DIM // size_per_thread_fastest_dim), WARP_SIZE), 1
        )
        self.Q_ROPE_LOAD_LAYOUT = gl.constexpr(
            gl.BlockedLayout(
                size_per_thread=[1, size_per_thread_fastest_dim],
                threads_per_warp=[
                    WARP_SIZE // threads_per_warp_fastest_dim1,
                    threads_per_warp_fastest_dim1,
                ],
                warps_per_cta=[NUM_WARPS, 1],
                order=[1, 0],
            )
        )

        self.q_cache_modifier = gl.constexpr(".cg")
        self.kv_cache_modifier = gl.constexpr(".cg")


@aggregate
class MLAProgram:
    """Program state and core operations for the unified attention kernel."""

    cfg: MLAConfig

    q_lora: gl.tensor
    q_rope: gl.tensor
    kv_lora_shared: gl.shared_memory_descriptor
    k_rope_shared: gl.shared_memory_descriptor

    kv_buffer_ptr: gl.tensor
    output_ptr: gl.tensor
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

    kv_lora_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    k_rope_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    stride_kv_buffer_0: gl.tensor
    stride_kv_buffer_1: gl.tensor
    stride_kv_buffer_2: gl.tensor
    stride_kv_buffer_3: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q_lora,
        q_rope,
        kv_lora_shared,
        k_rope_shared,
        kv_buffer_ptr,
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
        kv_lora_desc,
        k_rope_desc,
        stride_kv_buffer_0,
        stride_kv_buffer_1,
        stride_kv_buffer_2,
        stride_kv_buffer_3,
    ):
        self.cfg = cfg
        self.q_lora = q_lora
        self.q_rope = q_rope
        self.kv_buffer_ptr = kv_buffer_ptr
        self.output_ptr = output_ptr
        self.segm_max_ptr = segm_max_ptr
        self.segm_expsum_ptr = segm_expsum_ptr
        self.kv_lora_shared = kv_lora_shared
        self.k_rope_shared = k_rope_shared
        self.kv_lora_desc = kv_lora_desc
        self.k_rope_desc = k_rope_desc
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
        self.stride_kv_buffer_0 = stride_kv_buffer_0
        self.stride_kv_buffer_1 = stride_kv_buffer_1
        self.stride_kv_buffer_2 = stride_kv_buffer_2
        self.stride_kv_buffer_3 = stride_kv_buffer_3

    @gluon.jit
    def initialize(
        cfg: MLAConfig,
        q_lora,
        q_rope,
        kv_buffer_ptr,
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
        stride_kv_buffer_0,
        stride_kv_buffer_1,
        stride_kv_buffer_2,
        stride_kv_buffer_3,
    ):
        # cfg.SHUFFLED_KV_CACHE == 1
        # cfg.NUM_BLOCKS_GATHER_PER_TILE == 1:
        kv_lora_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=kv_buffer_ptr,
            shape=(
                num_blocks * cfg.NUM_KV_HEADS * (cfg.BLOCK_SIZE // 16),
                cfg.KV_LORA_RANK * 16,
            ),
            strides=(stride_kv_buffer_2, 1),
            block_shape=(cfg.BLOCK_SIZE // 16, cfg.KV_LORA_RANK * 16),
            layout=cfg.KV_LORA_SHARED_LAYOUT,
        )
        k_rope_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=kv_buffer_ptr + (cfg.KV_LORA_RANK * 16),
            shape=(
                num_blocks * cfg.NUM_KV_HEADS * (cfg.BLOCK_SIZE // 16),
                cfg.QK_ROPE_HEAD_DIM * 16,
            ),
            strides=(stride_kv_buffer_2, 1),
            block_shape=(cfg.BLOCK_SIZE // 16, cfg.QK_ROPE_HEAD_DIM * 16),
            layout=cfg.K_ROPE_SHARED_LAYOUT,
        )

        kv_lora_shared = gl.allocate_shared_memory(
            kv_lora_desc.dtype,
            [cfg.NUM_STAGES] + kv_lora_desc.block_shape,
            layout=cfg.KV_LORA_SHARED_LAYOUT,
        )
        k_rope_shared = gl.allocate_shared_memory(
            k_rope_desc.dtype,
            [cfg.NUM_STAGES] + k_rope_desc.block_shape,
            layout=cfg.K_ROPE_SHARED_LAYOUT,
        )

        # Calculate tile range
        num_tiles = (max_seq_prefix_len + cfg.BLOCK_SIZE - 1) // cfg.BLOCK_SIZE
        tile_start = segm_idx * tiles_per_segment
        tile_end = min((segm_idx + 1) * tiles_per_segment, num_tiles)

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

        return MLAProgram(
            cfg,
            q_lora,
            q_rope,
            kv_lora_shared,
            k_rope_shared,
            kv_buffer_ptr,
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
            kv_lora_desc,
            k_rope_desc,
            stride_kv_buffer_0,
            stride_kv_buffer_1,
            stride_kv_buffer_2,
            stride_kv_buffer_3,
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
    ):
        M = gl.full(
            [self.cfg.BLOCK_M],
            float("-inf"),
            dtype=gl.float32,
            layout=gl.SliceLayout(1, self.cfg.QK_WMMA_LAYOUT),
        )
        L = gl.full(
            [self.cfg.BLOCK_M],
            1.0,
            dtype=gl.float32,
            layout=gl.SliceLayout(1, self.cfg.QK_WMMA_LAYOUT),
        )
        acc = gl.zeros(
            [self.cfg.BLOCK_M, self.cfg.KV_LORA_RANK],
            dtype=gl.float32,
            layout=self.cfg.PV_WMMA_LAYOUT,
        )

        return L, M, acc

    @gluon.jit
    def load_physical_block_idx(self, j, block_tables_ptr_shifted):
        physical_block_idx = gl.load(block_tables_ptr_shifted + j)

        return j + 1, physical_block_idx

    @gluon.jit
    def lds_unshuffle_kv_lora(self, buffer_id):
        return (
            self.kv_lora_shared.index(buffer_id)
            .reshape(
                (
                    self.cfg.NUM_BLOCKS_GATHER_PER_TILE,
                    self.cfg.BLOCK_SIZE // 16,
                    self.cfg.KV_LORA_RANK // (2 * self.cfg.K_WIDTH),
                    2,
                    16,
                    self.cfg.K_WIDTH,
                )
            )
            .permute((0, 1, 4, 2, 3, 5))
            .reshape((self.cfg.TILE_SIZE, self.cfg.KV_LORA_RANK))
            .permute((1, 0))
        )

    @gluon.jit
    def lds_unshuffle_k_rope(self, buffer_id):
        return (
            self.k_rope_shared.index(buffer_id)
            .reshape(
                (
                    self.cfg.NUM_BLOCKS_GATHER_PER_TILE,
                    self.cfg.BLOCK_SIZE // 16,
                    self.cfg.QK_ROPE_HEAD_DIM // (2 * self.cfg.K_WIDTH),
                    2,
                    16,
                    self.cfg.K_WIDTH,
                )
            )
            .permute((0, 1, 4, 2, 3, 5))
            .reshape((self.cfg.TILE_SIZE, self.cfg.QK_ROPE_HEAD_DIM))
            .permute((1, 0))
        )

    @gluon.jit
    def tdm_shared_load_kv_lora(self, wait_count, buffer_id):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        return self.lds_unshuffle_kv_lora(buffer_id).load(layout=self.cfg.K_DOT_LAYOUT)

    @gluon.jit
    def tdm_shared_load_k_rope(self, wait_count, buffer_id):
        gl.amd.gfx1250.tdm.async_wait(wait_count)
        return self.lds_unshuffle_k_rope(buffer_id).load(layout=self.cfg.K_DOT_LAYOUT)

    @gluon.jit
    def get_kv_buffer_row_offsets(self, block_idx):
        return (
            (block_idx * self.cfg.NUM_KV_HEADS + self.kv_head_idx)
            * (self.cfg.BLOCK_SIZE // 16)
        ).to(gl.int32)

    @gluon.jit
    def tdm_load_global_to_shared_kv_lora(self, row_offsets, buffer_id):
        gl.amd.gfx1250.tdm.async_load(
            self.kv_lora_desc, [row_offsets, 0], self.kv_lora_shared.index(buffer_id)
        )

    @gluon.jit
    def tdm_load_global_to_shared_k_rope(self, row_offsets, buffer_id):
        gl.amd.gfx1250.tdm.async_load(
            self.k_rope_desc, [row_offsets, 0], self.k_rope_shared.index(buffer_id)
        )

    @gluon.jit
    def compute_qk_lora(self, k_lora, S):
        if self.cfg.IS_KV_FP8:
            k_lora = k_lora.to(self.q_lora.dtype)
        return gl.amd.gfx1250.wmma(self.q_lora, k_lora, S)

    @gluon.jit
    def compute_qk_rope(self, k_rope, S):
        if self.cfg.IS_KV_FP8:
            k_rope = k_rope.to(self.q_rope.dtype)
        return gl.amd.gfx1250.wmma(self.q_rope, k_rope, S)

    @gluon.jit
    def apply_mask_qk_3D(self, S, seq_offset):
        seq_mask = seq_offset[None, :] < self.context_len + self.query_pos_qk + 1
        S = gl.where(self.query_mask_qk & seq_mask, S, float("-inf"))
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
    def compute_pkv_lora_trans(self, p, kv_lora_trans, acc):
        if self.cfg.IS_KV_FP8:
            p = p.to(kv_lora_trans.dtype)
        else:
            p = p.to(gl.bfloat16, fp_downcast_rounding="rtz")
        p = gl.convert_layout(p, self.cfg.P_DOT_LAYOUT)
        acc = gl.amd.gfx1250.wmma(p, kv_lora_trans, acc)
        return acc

    @gluon.jit
    def store_output_3D(self, acc, M, L, segm_idx):
        # acc = gl.convert_layout(acc, layout=self.cfg.PV_WMMA_LAYOUT)
        offs_q_d_lora_pv = gl.arange(
            0, self.cfg.KV_LORA_RANK, layout=gl.SliceLayout(0, self.cfg.PV_WMMA_LAYOUT)
        )

        segm_output_offset = (
            self.query_offset_0_pv[:, None]
            * (
                self.cfg.NUM_QUERY_HEADS
                * self.cfg.NUM_SEGMENTS_PER_SEQ
                * self.cfg.KV_LORA_RANK
            )
            + self.query_offset_1_pv[:, None]
            * (self.cfg.NUM_SEGMENTS_PER_SEQ * self.cfg.KV_LORA_RANK)
            + segm_idx * self.cfg.KV_LORA_RANK
            + offs_q_d_lora_pv[None, :]
        )
        if self.cfg.USE_STORE_BUFFER_OP:
            gl.amd.cdna4.buffer_store(
                stored_value=acc,
                ptr=self.output_ptr,
                offsets=segm_output_offset,
                mask=self.query_mask_0_pv[:, None] & self.query_mask_1_pv[:, None],
            )
        else:
            gl.store(
                self.output_ptr + segm_output_offset.to(gl.int64),
                acc,
                mask=self.query_mask_0_pv[:, None] & self.query_mask_1_pv[:, None],
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
def fast_exp(x):
    RCP_LN2: gl.constexpr = 1.4426950408889634
    return gl.math.exp2(x * RCP_LN2)


@gluon.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@gluon.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = gl.math.exp2(Sdiv)
    p2 = gl.math.exp2(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@gluon.jit
def _find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: gl.constexpr,
    use_q_block_mode: gl.constexpr,
):
    left: gl.int32 = 0
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


_mla_decode_fwd_kernel_repr = make_kernel_repr(
    "_mla_decode_fwd_kernel",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "num_tokens_per_seq",
        "TILE_SIZE",
        "KV_LORA_RANK",
        "QK_ROPE_HEAD_DIM",
        "BLOCK_Q",
        "BLOCK_M",
        "NUM_SEGMENTS_PER_SEQ",
        "num_warps",
        "num_stages",
    ],
)


@gluon.jit(repr=_mla_decode_fwd_kernel_repr)
def _mla_decode_fwd_kernel(
    segm_output_ptr,  # [total_num_tokens, num_query_heads, KV_LORA_RANK + qk_rope_head_dim]
    segm_max_ptr,  # [total_num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [total_num_tokens, num_query_heads, num_segments]
    query_ptr,  # [total_num_tokens, num_query_heads, head_size]
    kv_buffer_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    SCALE: gl.constexpr,  # float32
    q_scale_ptr,  # float32
    kv_scale_ptr,  # float32
    num_query_heads: gl.constexpr,  # int
    num_kv_heads: gl.constexpr,  # int
    block_tables_stride: gl.int64,  # int
    query_stride_0: gl.int64,  # int
    query_stride_1: gl.int64,  # int, should be equal to head_size
    KV_LORA_RANK: gl.constexpr,  # int
    QK_ROPE_HEAD_DIM: gl.constexpr,  # int
    stride_kv_buffer_0: gl.int32,  # int
    stride_kv_buffer_1: gl.int32,  # int
    stride_kv_buffer_2: gl.int32,  # int
    stride_kv_buffer_3: gl.int32,  # int
    query_start_len_ptr,  # [num_seqs+1]
    num_tokens_per_seq: gl.int32,
    num_blocks: gl.int32,
    TILE_SIZE: gl.constexpr,  # int
    BLOCK_Q: gl.constexpr,  # int
    BLOCK_M: gl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: gl.constexpr,  # int
    WARP_SIZE: gl.constexpr,  # int
    num_warps: gl.constexpr,  # int
    num_stages: gl.constexpr,  # int
    SHUFFLED_KV_CACHE: gl.constexpr = True,  # bool
    ALL_DECODE: gl.constexpr = False,  # bool
    IS_Q_FP8: gl.constexpr = False,  # bool
    IS_KV_FP8: gl.constexpr = False,  # bool
):
    assert SHUFFLED_KV_CACHE == True
    cfg = MLAConfig(
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        TILE_SIZE,
        1,
        NUM_SEGMENTS_PER_SEQ,
        BLOCK_M,
        BLOCK_Q,
        num_query_heads,
        num_kv_heads,
        num_warps,
        WARP_SIZE,
        num_stages,
        SCALE,
        False,
        False,
        SHUFFLED_KV_CACHE,
        IS_Q_FP8,
        IS_KV_FP8,
    )
    q_block_global_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    segm_idx = gl.program_id(2)

    num_q_blocks_per_seq = cdiv_fn(num_tokens_per_seq, BLOCK_Q)

    if ALL_DECODE:
        seq_idx = q_block_global_idx
    else:
        seq_idx = q_block_global_idx // num_q_blocks_per_seq

    q_start_idx = gl.load(query_start_len_ptr + seq_idx)
    q_block_local_idx = q_block_global_idx - seq_idx * num_q_blocks_per_seq

    # sequence len for this particular sequence
    seq_len = gl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    q_lora_shared = gl.allocate_shared_memory(
        query_ptr.type.element_ty,
        shape=[BLOCK_M, KV_LORA_RANK],
        layout=cfg.Q_LORA_SHARED_LAYOUT,
    )
    q_rope_shared = gl.allocate_shared_memory(
        query_ptr.type.element_ty,
        shape=[BLOCK_M, QK_ROPE_HEAD_DIM],
        layout=cfg.Q_ROPE_SHARED_LAYOUT,
    )

    qk_factor: gl.float32 = cfg.QK_SCALE
    if q_scale_ptr is not None:
        q_scale = gl.load(q_scale_ptr)
        qk_factor = qk_factor * q_scale
    else:
        q_scale = None

    if kv_scale_ptr is not None:
        kv_scale = gl.load(kv_scale_ptr)
        qk_factor = qk_factor * kv_scale
    else:
        kv_scale = None

    offs_q_m_lora = gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, cfg.Q_LORA_LOAD_LAYOUT)
    )
    offs_q_d_lora = gl.arange(
        0, KV_LORA_RANK, layout=gl.SliceLayout(0, cfg.Q_LORA_LOAD_LAYOUT)
    )
    offs_q_m_rope = gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, cfg.Q_ROPE_LOAD_LAYOUT)
    )
    offs_q_d_rope = gl.arange(
        0, QK_ROPE_HEAD_DIM, layout=gl.SliceLayout(0, cfg.Q_ROPE_LOAD_LAYOUT)
    )
    # KV_LORA_LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread=[8, 1],
    #     threads_per_warp=[32, 1],
    #     warps_per_cta=[1, num_warps],
    #     order=[0, 1],
    # )
    # K_ROPE_LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread=[8, 1],
    #     threads_per_warp=[8, 4],
    #     warps_per_cta=[1, num_warps],
    #     order=[0, 1],
    # )
    # offs_kv_t_lora = gl.arange(
    #     0, TILE_SIZE, layout=gl.SliceLayout(0, KV_LORA_LOAD_LAYOUT)
    # )
    # offs_kv_d_lora = gl.arange(
    #     0, KV_LORA_RANK, layout=gl.SliceLayout(1, KV_LORA_LOAD_LAYOUT)
    # )
    # offs_k_t_rope = gl.arange(
    #     0, TILE_SIZE, layout=gl.SliceLayout(0, K_ROPE_LOAD_LAYOUT)
    # )
    # offs_k_d_rope = gl.arange(
    #     0, QK_ROPE_HEAD_DIM, layout=gl.SliceLayout(1, K_ROPE_LOAD_LAYOUT)
    # )

    query_pos_lora = (
        q_block_local_idx * BLOCK_Q + offs_q_m_lora // cfg.NUM_QUERIES_PER_KV
    )
    query_offset_0_lora = q_start_idx + query_pos_lora
    query_offset_1_lora = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_lora % cfg.NUM_QUERIES_PER_KV
    )
    query_offset_lora = (
        query_offset_0_lora[:, None] * query_stride_0
        + query_offset_1_lora[:, None] * query_stride_1
    )
    query_mask_0_lora = query_pos_lora < num_tokens_per_seq
    query_mask_1_lora = query_offset_1_lora < num_query_heads

    # Q_lora : (BLOCK_M, KV_LORA_RANK)
    Q_lora_load = gl.load(
        query_ptr + query_offset_lora + offs_q_d_lora[None, :],
        mask=query_mask_0_lora[:, None] & query_mask_1_lora[:, None],
        other=0.0,
    )
    q_lora_shared.store(Q_lora_load)
    Q_lora = q_lora_shared.load(layout=cfg.Q_DOT_LAYOUT)

    query_pos_rope = (
        q_block_local_idx * BLOCK_Q + offs_q_m_rope // cfg.NUM_QUERIES_PER_KV
    )
    query_offset_0_rope = q_start_idx + query_pos_rope
    query_offset_1_rope = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_rope % cfg.NUM_QUERIES_PER_KV
    )
    query_offset_rope = (
        query_offset_0_rope[:, None] * query_stride_0
        + query_offset_1_rope[:, None] * query_stride_1
    )
    query_mask_0_rope = query_pos_rope < num_tokens_per_seq
    query_mask_1_rope = query_offset_1_rope < num_query_heads

    # Q_rope : (BLOCK_M, QK_ROPE_HEAD_DIM)
    Q_rope_load = gl.load(
        query_ptr + query_offset_rope + (KV_LORA_RANK + offs_q_d_rope)[None, :],
        mask=query_mask_0_rope[:, None] & query_mask_1_rope[:, None],
        other=0.0,
    )
    q_rope_shared.store(Q_rope_load)
    Q_rope = q_rope_shared.load(layout=cfg.Q_DOT_LAYOUT)

    offs_q_m_qk = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.QK_WMMA_LAYOUT))
    query_pos_qk = q_block_local_idx * BLOCK_Q + offs_q_m_qk // cfg.NUM_QUERIES_PER_KV
    query_offset_0_qk = q_start_idx + query_pos_qk
    query_offset_1_qk = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_qk % cfg.NUM_QUERIES_PER_KV
    )
    query_mask_0_qk = query_pos_qk < num_tokens_per_seq
    query_mask_1_qk = query_offset_1_qk < num_query_heads
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

    block_tables_ptr_shifted = block_tables_ptr + seq_idx * block_tables_stride

    # context length for this particular sequences
    context_len = seq_len - num_tokens_per_seq

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // cfg.NUM_QUERIES_PER_KV
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)

    pgm: MLAProgram = MLAProgram.initialize(
        cfg,
        Q_lora,
        Q_rope,
        kv_buffer_ptr,
        segm_output_ptr,
        segm_max_ptr,
        segm_expsum_ptr,
        max_seq_prefix_len,
        q_block_local_idx,
        num_q_blocks_per_seq,
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
        stride_kv_buffer_0,
        stride_kv_buffer_1,
        stride_kv_buffer_2,
        stride_kv_buffer_3,
    )

    L, M, acc = pgm.allocate_accumulator()

    j_from_hbm: gl.int32 = segm_idx * tiles_per_segment
    buffer_id: gl.int32 = 0
    seq_offset = j_from_hbm * cfg.TILE_SIZE + gl.arange(
        0, cfg.TILE_SIZE, layout=gl.SliceLayout(0, cfg.QK_WMMA_LAYOUT)
    )

    for _ in range(cfg.NUM_STAGES - 1):
        j_from_hbm, physical_block_idx = pgm.load_physical_block_idx(
            j_from_hbm, block_tables_ptr_shifted
        )
        row_offsets = pgm.get_kv_buffer_row_offsets(physical_block_idx)
        pgm.tdm_load_global_to_shared_kv_lora(row_offsets, buffer_id=buffer_id)
        pgm.tdm_load_global_to_shared_k_rope(row_offsets, buffer_id=buffer_id)

    for _ in range(pgm.tile_start, pgm.tile_end - (cfg.NUM_STAGES - 1)):
        j_from_hbm, physical_block_idx = pgm.load_physical_block_idx(
            j_from_hbm, block_tables_ptr_shifted
        )

        kv_lora = pgm.tdm_shared_load_kv_lora(
            wait_count=(cfg.NUM_STAGES - 2) * 2 + 1, buffer_id=buffer_id
        )
        next_buffer_id = pgm.get_next_buffer_id(buffer_id)
        row_offsets = pgm.get_kv_buffer_row_offsets(physical_block_idx)
        pgm.tdm_load_global_to_shared_kv_lora(row_offsets, buffer_id=next_buffer_id)
        pgm.tdm_load_global_to_shared_k_rope(row_offsets, buffer_id=next_buffer_id)

        k_rope = pgm.tdm_shared_load_k_rope(
            wait_count=(cfg.NUM_STAGES - 1) * 2, buffer_id=buffer_id
        )

        S = gl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32, layout=cfg.QK_WMMA_LAYOUT)
        S = pgm.compute_qk_lora(kv_lora, S)
        S = pgm.compute_qk_rope(k_rope, S)

        S = S * qk_factor

        S = pgm.apply_mask_qk_3D(S, seq_offset)

        p, alpha, M = pgm.softmax_part0(S, M)
        p, L, acc = pgm.softmax_part1(p, L, acc, alpha)

        kv_lora_trans = kv_lora.permute((1, 0))
        kv_lora_trans = gl.convert_layout(kv_lora_trans, layout=cfg.V_DOT_LAYOUT)
        acc = pgm.compute_pkv_lora_trans(p, kv_lora_trans, acc)

        buffer_id = next_buffer_id
        seq_offset += cfg.TILE_SIZE

    for _ in range(cfg.NUM_STAGES - 1):
        kv_lora = pgm.tdm_shared_load_kv_lora(
            wait_count=(cfg.NUM_STAGES - 2) * 2 + 1, buffer_id=buffer_id
        )
        k_rope = pgm.tdm_shared_load_k_rope(
            wait_count=(cfg.NUM_STAGES - 2) * 2, buffer_id=buffer_id
        )
        S = gl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32, layout=cfg.QK_WMMA_LAYOUT)
        S = pgm.compute_qk_lora(kv_lora, S)
        S = pgm.compute_qk_rope(k_rope, S)
        S = S * qk_factor

        p, alpha, M = pgm.softmax_part0(S, M)
        p, L, acc = pgm.softmax_part1(p, L, acc, alpha)

        kv_lora_trans = kv_lora.permute((1, 0))
        kv_lora_trans = gl.convert_layout(kv_lora_trans, layout=cfg.V_DOT_LAYOUT)
        acc = pgm.compute_pkv_lora_trans(p, kv_lora_trans, acc)

        if cfg.NUM_STAGES > 2:
            buffer_id = pgm.get_next_buffer_id(buffer_id)
            seq_offset += cfg.TILE_SIZE

    if kv_scale_ptr is not None:
        acc = acc * kv_scale

    pgm.store_output_3D(
        acc,
        M,
        L,
        segm_idx,
    )


_mla_prefill_fwd_kernel_non_pipelined_repr = make_kernel_repr(
    "_mla_prefill_fwd_kernel_non_pipelined",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "num_tokens_per_seq",
        "TILE_SIZE",
        "KV_LORA_RANK",
        "QK_ROPE_HEAD_DIM",
        "BLOCK_Q",
        "BLOCK_M",
        "NUM_SEGMENTS_PER_SEQ",
        "num_warps",
        "num_stages",
    ],
)


@gluon.jit(repr=_mla_prefill_fwd_kernel_non_pipelined_repr)
def _mla_prefill_fwd_kernel_non_pipelined(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    kv_buffer_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    SCALE: gl.constexpr,  # float32
    q_scale_ptr,  # float32
    kv_scale_ptr,  # float32
    out_scale_ptr,  # float32
    num_query_heads: gl.constexpr,  # int
    num_kv_heads: gl.constexpr,  # int
    block_tables_stride: gl.int64,  # int
    query_stride_0: gl.int64,  # int
    query_stride_1: gl.int64,  # int, should be equal to head_size
    output_stride_0: gl.int64,  # int
    output_stride_1: gl.int64,  # int, should be equal to head_size
    KV_LORA_RANK: gl.constexpr,  # int
    QK_ROPE_HEAD_DIM: gl.constexpr,  # int
    stride_kv_buffer_0: gl.int64,  # int
    stride_kv_buffer_1: gl.int64,  # int
    stride_kv_buffer_2: gl.int64,  # int
    stride_kv_buffer_3: gl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    num_seqs: gl.int32,
    TILE_SIZE: gl.constexpr,  # int
    BLOCK_Q: gl.constexpr,  # int
    BLOCK_M: gl.constexpr,  # int
    WARP_SIZE: gl.constexpr,  # int
    num_warps: gl.constexpr,  # int
    num_stages: gl.constexpr,  # int
    IS_Q_FP8: gl.constexpr = False,  # bool
    IS_KV_FP8: gl.constexpr = False,  # bool
    FP8_MIN: gl.constexpr = float8_info.min,
    FP8_MAX: gl.constexpr = float8_info.max,
):
    cfg = MLAConfig(
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        TILE_SIZE,
        1,
        1,
        BLOCK_M,
        BLOCK_Q,
        num_query_heads,
        num_kv_heads,
        num_warps,
        WARP_SIZE,
        num_stages,
        SCALE,
        False,
        False,
        False,
        IS_Q_FP8,
        IS_KV_FP8,
    )
    kv_head_idx = gl.program_id(0)
    q_block_global_idx = gl.program_id(1)

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = gl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = gl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = gl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    # gl.device_print("cur_batch_query_len", cur_batch_query_len)
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    q_lora_shared = gl.allocate_shared_memory(
        query_ptr.type.element_ty,
        shape=[BLOCK_M, KV_LORA_RANK],
        layout=cfg.Q_LORA_SHARED_LAYOUT,
    )
    q_rope_shared = gl.allocate_shared_memory(
        query_ptr.type.element_ty,
        shape=[BLOCK_M, QK_ROPE_HEAD_DIM],
        layout=cfg.Q_ROPE_SHARED_LAYOUT,
    )
    kv_lora_shared = gl.allocate_shared_memory(
        kv_buffer_ptr.type.element_ty,
        [KV_LORA_RANK, TILE_SIZE],
        layout=cfg.KV_LORA_SHARED_LAYOUT,
    )
    k_rope_shared = gl.allocate_shared_memory(
        kv_buffer_ptr.type.element_ty,
        [QK_ROPE_HEAD_DIM, TILE_SIZE],
        layout=cfg.K_ROPE_SHARED_LAYOUT,
    )

    qk_factor: gl.float32 = cfg.QK_SCALE
    if q_scale_ptr is not None:
        q_scale = gl.load(q_scale_ptr)
        qk_factor = qk_factor * q_scale
    else:
        q_scale = None

    if kv_scale_ptr is not None:
        kv_scale = gl.load(kv_scale_ptr)
        qk_factor = qk_factor * kv_scale
    else:
        kv_scale = None
    out_scale = None
    if out_scale_ptr is not None:
        out_scale = 1 / gl.load(out_scale_ptr)

    offs_q_m_lora = gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, cfg.Q_LORA_LOAD_LAYOUT)
    )
    offs_q_m_rope = gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, cfg.Q_ROPE_LOAD_LAYOUT)
    )
    offs_q_d_lora = gl.arange(
        0, KV_LORA_RANK, layout=gl.SliceLayout(0, cfg.Q_LORA_LOAD_LAYOUT)
    )
    offs_q_d_rope = gl.arange(
        0, QK_ROPE_HEAD_DIM, layout=gl.SliceLayout(0, cfg.Q_ROPE_LOAD_LAYOUT)
    )
    KV_LORA_LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[32, 1],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )
    K_ROPE_LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[8, 4],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )
    offs_kv_t_lora = gl.arange(
        0, TILE_SIZE, layout=gl.SliceLayout(0, KV_LORA_LOAD_LAYOUT)
    )
    offs_kv_d_lora = gl.arange(
        0, KV_LORA_RANK, layout=gl.SliceLayout(1, KV_LORA_LOAD_LAYOUT)
    )
    offs_k_t_rope = gl.arange(
        0, TILE_SIZE, layout=gl.SliceLayout(0, K_ROPE_LOAD_LAYOUT)
    )
    offs_k_d_rope = gl.arange(
        0, QK_ROPE_HEAD_DIM, layout=gl.SliceLayout(1, K_ROPE_LOAD_LAYOUT)
    )

    query_pos_lora = (
        q_block_local_idx * BLOCK_Q + offs_q_m_lora // cfg.NUM_QUERIES_PER_KV
    )

    query_offset_0_lora = cur_batch_in_all_start_index + query_pos_lora
    query_offset_1_lora = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_lora % cfg.NUM_QUERIES_PER_KV
    )
    query_offset_lora = (
        query_offset_0_lora[:, None] * query_stride_0
        + query_offset_1_lora[:, None] * query_stride_1
    )
    query_mask_0_lora = query_pos_lora < cur_batch_query_len
    query_mask_1_lora = query_offset_1_lora < num_query_heads

    # Q_lora : (BLOCK_M, KV_LORA_RANK)
    Q_lora_load = gl.load(
        query_ptr + query_offset_lora + offs_q_d_lora[None, :],
        mask=query_mask_0_lora[:, None] & query_mask_1_lora[:, None],
        other=0.0,
    )
    q_lora_shared.store(Q_lora_load)
    Q_lora = q_lora_shared.load(layout=cfg.Q_DOT_LAYOUT)

    query_pos_rope = (
        q_block_local_idx * BLOCK_Q + offs_q_m_rope // cfg.NUM_QUERIES_PER_KV
    )
    query_offset_0_rope = cur_batch_in_all_start_index + query_pos_rope
    query_offset_1_rope = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_rope % cfg.NUM_QUERIES_PER_KV
    )
    query_offset_rope = (
        query_offset_0_rope[:, None] * query_stride_0
        + query_offset_1_rope[:, None] * query_stride_1
    )
    query_mask_0_rope = query_pos_rope < cur_batch_query_len
    query_mask_1_rope = query_offset_1_rope < num_query_heads

    # Q_rope : (BLOCK_M, QK_ROPE_HEAD_DIM)
    Q_rope_load = gl.load(
        query_ptr + query_offset_rope + (KV_LORA_RANK + offs_q_d_rope)[None, :],
        mask=query_mask_0_rope[:, None] & query_mask_1_rope[:, None],
        other=0.0,
    )
    q_rope_shared.store(Q_rope_load)
    Q_rope = q_rope_shared.load(layout=cfg.Q_DOT_LAYOUT)

    offs_q_m_qk = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.QK_WMMA_LAYOUT))
    query_pos_qk = q_block_local_idx * BLOCK_Q + offs_q_m_qk // cfg.NUM_QUERIES_PER_KV
    query_offset_0_qk = cur_batch_in_all_start_index + query_pos_qk
    query_offset_1_qk = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_qk % cfg.NUM_QUERIES_PER_KV
    )
    query_offset_qk = (
        query_offset_0_qk[:, None] * query_stride_0
        + query_offset_1_qk[:, None] * query_stride_1
    )
    query_mask_0_qk = query_pos_qk < cur_batch_query_len
    query_mask_1_qk = query_offset_1_qk < num_query_heads
    offs_seq_t = gl.arange(0, TILE_SIZE, layout=gl.SliceLayout(0, cfg.QK_WMMA_LAYOUT))

    block_tables_ptr_shifted = block_tables_ptr + seq_idx * block_tables_stride

    M = gl.full(
        [BLOCK_M],
        float("-inf"),
        dtype=gl.float32,
        layout=gl.SliceLayout(1, cfg.QK_WMMA_LAYOUT),
    )
    L = gl.full(
        [BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, cfg.QK_WMMA_LAYOUT)
    )
    acc = gl.zeros([BLOCK_M, KV_LORA_RANK], dtype=gl.float32, layout=cfg.PV_WMMA_LAYOUT)

    # sequence len for this particular sequence
    seq_len = gl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // cfg.NUM_QUERIES_PER_KV
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    seq_offset = offs_seq_t

    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        physical_block_idx = gl.load(block_tables_ptr_shifted + j).to(gl.int64)

        kv_offset = (
            physical_block_idx * stride_kv_buffer_0 + kv_head_idx * stride_kv_buffer_2
        )

        kv_lora_offset = (
            kv_offset
            + offs_kv_t_lora[None, :] * stride_kv_buffer_1
            + offs_kv_d_lora[:, None] * stride_kv_buffer_3
        )
        # KV_lora : (BLOCK_M, KV_LORA_RANK)
        KV_lora_load = gl.load(
            kv_buffer_ptr + kv_lora_offset,
            cache_modifier=cfg.kv_cache_modifier,
        )
        kv_lora_shared.store(KV_lora_load)

        k_rope_offset = (
            kv_offset
            + offs_k_t_rope[None, :] * stride_kv_buffer_1
            + (KV_LORA_RANK + offs_k_d_rope)[:, None] * stride_kv_buffer_3
        )
        # K_rope : (BLOCK_M, QK_ROPE_HEAD_DIM)
        K_rope_load = gl.load(
            kv_buffer_ptr + k_rope_offset,
            cache_modifier=cfg.kv_cache_modifier,
        )
        k_rope_shared.store(K_rope_load)

        KV_lora = kv_lora_shared.load(layout=cfg.K_DOT_LAYOUT)
        S = gl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32, layout=cfg.QK_WMMA_LAYOUT)
        S = gl.amd.gfx1250.wmma(Q_lora, KV_lora.to(Q_lora.dtype), S)
        K_rope = k_rope_shared.load(layout=cfg.K_DOT_LAYOUT)
        S = gl.amd.gfx1250.wmma(Q_rope, K_rope.to(Q_lora.dtype), S) * qk_factor

        seq_mask = seq_offset[None, :] < context_len + query_pos_qk[:, None] + 1

        S = gl.where(
            query_mask_1_qk[:, None] & query_mask_0_qk[:, None] & seq_mask,
            S,
            float("-inf"),
        )

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = gl.maximum(M, gl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = gl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = gl.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = gl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = gl.exp2(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * gl.convert_layout(alpha[:, None], layout=cfg.PV_WMMA_LAYOUT)

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, KV_LORA_RANK)
        KV_lora_trans = kv_lora_shared.permute((1, 0)).load(layout=cfg.V_DOT_LAYOUT)
        P = P.to(KV_lora_trans.dtype)
        P = gl.convert_layout(P, layout=cfg.P_DOT_LAYOUT)
        acc = gl.amd.gfx1250.wmma(P, KV_lora_trans, acc)
        seq_offset += TILE_SIZE

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    if kv_scale_ptr is not None:
        one_over_L = kv_scale / L[:, None]
    else:
        one_over_L = 1.0 / L[:, None]
    acc = acc * gl.convert_layout(one_over_L, layout=cfg.PV_WMMA_LAYOUT)

    if out_scale_ptr is not None:
        acc = acc * out_scale
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)  # gluon has no clamp interface

    offs_q_m_pv = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT))
    offs_q_d_lora_pv = gl.arange(
        0, KV_LORA_RANK, layout=gl.SliceLayout(0, cfg.PV_WMMA_LAYOUT)
    )
    query_pos_pv = q_block_local_idx * BLOCK_Q + offs_q_m_pv // cfg.NUM_QUERIES_PER_KV
    query_offset_0_pv = cur_batch_in_all_start_index + query_pos_pv
    query_offset_1_pv = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_pv % cfg.NUM_QUERIES_PER_KV
    )
    query_mask_0_pv = query_pos_pv < cur_batch_query_len
    query_mask_1_pv = query_offset_1_pv < num_query_heads

    output_offset = (
        query_offset_0_pv[:, None] * output_stride_0
        + query_offset_1_pv[:, None] * output_stride_1
        + offs_q_d_lora_pv[None, :]
    )

    gl.store(
        output_ptr + output_offset,
        acc,
        mask=query_mask_0_pv[:, None] & query_mask_1_pv[:, None],
    )


_mla_decode_fwd_kernel_non_pipelined_repr = make_kernel_repr(
    "_mla_decode_fwd_kernel_non_pipelined",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "num_tokens_per_seq",
        "TILE_SIZE",
        "KV_LORA_RANK",
        "QK_ROPE_HEAD_DIM",
        "BLOCK_Q",
        "BLOCK_M",
        "NUM_SEGMENTS_PER_SEQ",
        "num_warps",
        "num_stages",
    ],
)


@gluon.jit(repr=_mla_decode_fwd_kernel_non_pipelined_repr)
def _mla_decode_fwd_kernel_non_pipelined(
    segm_output_ptr,  # [total_num_tokens, num_query_heads, KV_LORA_RANK + qk_rope_head_dim]
    segm_max_ptr,  # [total_num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [total_num_tokens, num_query_heads, num_segments]
    query_ptr,  # [total_num_tokens, num_query_heads, head_size]
    kv_buffer_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    SCALE: gl.constexpr,  # float32
    q_scale_ptr,  # float32
    kv_scale_ptr,  # float32
    num_query_heads: gl.constexpr,  # int
    num_kv_heads: gl.constexpr,  # int
    block_tables_stride: gl.int64,  # int
    query_stride_0: gl.int64,  # int
    query_stride_1: gl.int64,  # int, should be equal to head_size
    KV_LORA_RANK: gl.constexpr,  # int
    QK_ROPE_HEAD_DIM: gl.constexpr,  # int
    stride_kv_buffer_0: gl.int64,  # int
    stride_kv_buffer_1: gl.int64,  # int
    stride_kv_buffer_2: gl.int64,  # int
    stride_kv_buffer_3: gl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    num_tokens_per_seq: gl.int32,
    num_blocks: gl.int32,
    TILE_SIZE: gl.constexpr,  # int
    BLOCK_Q: gl.constexpr,  # int
    BLOCK_M: gl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: gl.constexpr,  # int
    WARP_SIZE: gl.constexpr,  # int
    num_warps: gl.constexpr,  # int
    num_stages: gl.constexpr,  # int
    SHUFFLED_KV_CACHE: gl.constexpr = False,  # bool
    ALL_DECODE: gl.constexpr = False,  # bool
    IS_Q_FP8: gl.constexpr = False,  # bool
    IS_KV_FP8: gl.constexpr = False,  # bool
):
    assert SHUFFLED_KV_CACHE == False
    cfg = MLAConfig(
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        TILE_SIZE,
        1,
        NUM_SEGMENTS_PER_SEQ,
        BLOCK_M,
        BLOCK_Q,
        num_query_heads,
        num_kv_heads,
        num_warps,
        WARP_SIZE,
        num_stages,
        SCALE,
        False,
        False,
        SHUFFLED_KV_CACHE,
        IS_Q_FP8,
        IS_KV_FP8,
    )
    q_block_global_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    segm_idx = gl.program_id(2)

    num_q_blocks_per_seq = cdiv_fn(num_tokens_per_seq, BLOCK_Q)

    if ALL_DECODE:
        seq_idx = q_block_global_idx
    else:
        seq_idx = q_block_global_idx // num_q_blocks_per_seq

    q_start_idx = gl.load(query_start_len_ptr + seq_idx)
    q_block_local_idx = q_block_global_idx - seq_idx * num_q_blocks_per_seq

    # sequence len for this particular sequence
    seq_len = gl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    q_lora_shared = gl.allocate_shared_memory(
        query_ptr.type.element_ty,
        shape=[BLOCK_M, KV_LORA_RANK],
        layout=cfg.Q_LORA_SHARED_LAYOUT,
    )
    q_rope_shared = gl.allocate_shared_memory(
        query_ptr.type.element_ty,
        shape=[BLOCK_M, QK_ROPE_HEAD_DIM],
        layout=cfg.Q_ROPE_SHARED_LAYOUT,
    )
    kv_lora_shared = gl.allocate_shared_memory(
        kv_buffer_ptr.type.element_ty,
        [TILE_SIZE, KV_LORA_RANK],
        layout=cfg.KV_LORA_SHARED_LAYOUT,
    )
    k_rope_shared = gl.allocate_shared_memory(
        kv_buffer_ptr.type.element_ty,
        [TILE_SIZE, QK_ROPE_HEAD_DIM],
        layout=cfg.K_ROPE_SHARED_LAYOUT,
    )

    qk_factor: gl.float32 = cfg.QK_SCALE
    if q_scale_ptr is not None:
        q_scale = gl.load(q_scale_ptr)
        qk_factor = qk_factor * q_scale
    else:
        q_scale = None

    if kv_scale_ptr is not None:
        kv_scale = gl.load(kv_scale_ptr)
        qk_factor = qk_factor * kv_scale
    else:
        kv_scale = None

    offs_q_m_lora = gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, cfg.Q_LORA_LOAD_LAYOUT)
    )
    offs_q_d_lora = gl.arange(
        0, KV_LORA_RANK, layout=gl.SliceLayout(0, cfg.Q_LORA_LOAD_LAYOUT)
    )
    offs_q_m_rope = gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, cfg.Q_ROPE_LOAD_LAYOUT)
    )
    offs_q_d_rope = gl.arange(
        0, QK_ROPE_HEAD_DIM, layout=gl.SliceLayout(0, cfg.Q_ROPE_LOAD_LAYOUT)
    )
    KV_LORA_LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 32],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    K_ROPE_LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 8],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    offs_kv_t_lora = gl.arange(
        0, TILE_SIZE, layout=gl.SliceLayout(1, KV_LORA_LOAD_LAYOUT)
    )
    offs_kv_d_lora = gl.arange(
        0, KV_LORA_RANK, layout=gl.SliceLayout(0, KV_LORA_LOAD_LAYOUT)
    )
    offs_k_t_rope = gl.arange(
        0, TILE_SIZE, layout=gl.SliceLayout(1, K_ROPE_LOAD_LAYOUT)
    )
    offs_k_d_rope = gl.arange(
        0, QK_ROPE_HEAD_DIM, layout=gl.SliceLayout(0, K_ROPE_LOAD_LAYOUT)
    )

    query_pos_lora = (
        q_block_local_idx * BLOCK_Q + offs_q_m_lora // cfg.NUM_QUERIES_PER_KV
    )
    query_offset_0_lora = q_start_idx + query_pos_lora
    query_offset_1_lora = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_lora % cfg.NUM_QUERIES_PER_KV
    )
    query_offset_lora = (
        query_offset_0_lora[:, None] * query_stride_0
        + query_offset_1_lora[:, None] * query_stride_1
    )
    query_mask_0_lora = query_pos_lora < num_tokens_per_seq
    query_mask_1_lora = query_offset_1_lora < num_query_heads

    # Q_lora : (BLOCK_M, KV_LORA_RANK)
    Q_lora_load = gl.load(
        query_ptr + query_offset_lora + offs_q_d_lora[None, :],
        mask=query_mask_0_lora[:, None] & query_mask_1_lora[:, None],
        other=0.0,
    )
    q_lora_shared.store(Q_lora_load)
    Q_lora = q_lora_shared.load(layout=cfg.Q_DOT_LAYOUT)

    query_pos_rope = (
        q_block_local_idx * BLOCK_Q + offs_q_m_rope // cfg.NUM_QUERIES_PER_KV
    )
    query_offset_0_rope = q_start_idx + query_pos_rope
    query_offset_1_rope = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_rope % cfg.NUM_QUERIES_PER_KV
    )
    query_offset_rope = (
        query_offset_0_rope[:, None] * query_stride_0
        + query_offset_1_rope[:, None] * query_stride_1
    )
    query_mask_0_rope = query_pos_rope < num_tokens_per_seq
    query_mask_1_rope = query_offset_1_rope < num_query_heads

    # Q_rope : (BLOCK_M, QK_ROPE_HEAD_DIM)
    Q_rope_load = gl.load(
        query_ptr + query_offset_rope + (KV_LORA_RANK + offs_q_d_rope)[None, :],
        mask=query_mask_0_rope[:, None] & query_mask_1_rope[:, None],
        other=0.0,
    )
    q_rope_shared.store(Q_rope_load)
    Q_rope = q_rope_shared.load(layout=cfg.Q_DOT_LAYOUT)

    offs_q_m_qk = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.QK_WMMA_LAYOUT))
    query_pos_qk = q_block_local_idx * BLOCK_Q + offs_q_m_qk // cfg.NUM_QUERIES_PER_KV
    query_offset_0_qk = q_start_idx + query_pos_qk
    query_offset_1_qk = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_qk % cfg.NUM_QUERIES_PER_KV
    )
    query_mask_0_qk = query_pos_qk < num_tokens_per_seq
    query_mask_1_qk = query_offset_1_qk < num_query_heads
    offs_seq_t = gl.arange(0, TILE_SIZE, layout=gl.SliceLayout(0, cfg.QK_WMMA_LAYOUT))

    block_tables_ptr_shifted = block_tables_ptr + seq_idx * block_tables_stride

    M = gl.full(
        [BLOCK_M],
        float("-inf"),
        dtype=gl.float32,
        layout=gl.SliceLayout(1, cfg.QK_WMMA_LAYOUT),
    )
    L = gl.full(
        [BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, cfg.QK_WMMA_LAYOUT)
    )
    acc = gl.zeros([BLOCK_M, KV_LORA_RANK], dtype=gl.float32, layout=cfg.PV_WMMA_LAYOUT)

    # context length for this particular sequences
    context_len = seq_len - num_tokens_per_seq

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // cfg.NUM_QUERIES_PER_KV
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    seq_offset = segm_idx * tiles_per_segment * TILE_SIZE + offs_seq_t

    # iterate through tiles within current segment
    for j in range(
        segm_idx * tiles_per_segment,
        min((segm_idx + 1) * tiles_per_segment, num_tiles),
    ):
        physical_block_idx = gl.load(block_tables_ptr_shifted + j).to(gl.int64)

        kv_offset = (
            physical_block_idx * stride_kv_buffer_0 + kv_head_idx * stride_kv_buffer_2
        )

        kv_lora_offset = (
            kv_offset
            + offs_kv_t_lora[:, None] * stride_kv_buffer_1
            + offs_kv_d_lora[None, :] * stride_kv_buffer_3
        )
        # KV_lora : (BLOCK_M, KV_LORA_RANK)
        KV_lora_load = gl.load(
            kv_buffer_ptr + kv_lora_offset,
            cache_modifier=cfg.kv_cache_modifier,
        )
        kv_lora_shared.store(KV_lora_load)

        k_rope_offset = (
            kv_offset
            + offs_k_t_rope[:, None] * stride_kv_buffer_1
            + (KV_LORA_RANK + offs_k_d_rope[None, :]) * stride_kv_buffer_3
        )
        # K_rope : (BLOCK_M, QK_ROPE_HEAD_DIM)
        K_rope_load = gl.load(
            kv_buffer_ptr + k_rope_offset,
            cache_modifier=cfg.kv_cache_modifier,
        )
        k_rope_shared.store(K_rope_load)

        S = gl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32, layout=cfg.QK_WMMA_LAYOUT)

        KV_lora = kv_lora_shared.permute((1, 0)).load(layout=cfg.K_DOT_LAYOUT)
        S = gl.amd.gfx1250.wmma(Q_lora, KV_lora.to(Q_lora.dtype), S)
        K_rope = k_rope_shared.permute((1, 0)).load(layout=cfg.K_DOT_LAYOUT)
        S = gl.amd.gfx1250.wmma(Q_rope, K_rope.to(Q_lora.dtype), S) * qk_factor

        seq_mask = seq_offset[None, :] < context_len + query_pos_qk[:, None] + 1

        S = gl.where(
            query_mask_1_qk[:, None] & query_mask_0_qk[:, None] & seq_mask,
            S,
            float("-inf"),
        )

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = gl.maximum(M, gl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = gl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = gl.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = gl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = gl.exp2(M - m_j)

        # acc : (BLOCK_M, KV_LORA_RANK)
        acc = acc * gl.convert_layout(alpha[:, None], layout=cfg.PV_WMMA_LAYOUT)

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, KV_LORA_RANK)
        KV_lora_trans = kv_lora_shared.load(layout=cfg.V_DOT_LAYOUT)
        P = P.to(KV_lora_trans.dtype)
        P = gl.convert_layout(P, layout=cfg.P_DOT_LAYOUT)
        acc = gl.amd.gfx1250.wmma(P, KV_lora_trans, acc)
        seq_offset += TILE_SIZE

    if kv_scale_ptr is not None:
        acc = acc * kv_scale

    offs_q_m_pv = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT))
    offs_q_d_lora_pv = gl.arange(
        0, KV_LORA_RANK, layout=gl.SliceLayout(0, cfg.PV_WMMA_LAYOUT)
    )
    query_pos_pv = q_block_local_idx * BLOCK_Q + offs_q_m_pv // cfg.NUM_QUERIES_PER_KV
    query_offset_0_pv = q_start_idx + query_pos_pv
    query_offset_1_pv = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV + offs_q_m_pv % cfg.NUM_QUERIES_PER_KV
    )
    query_mask_0_pv = query_pos_pv < num_tokens_per_seq
    query_mask_1_pv = query_offset_1_pv < num_query_heads

    segm_output_offset = (
        query_offset_0_pv[:, None].to(gl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + query_offset_1_pv[:, None] * (NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + segm_idx * KV_LORA_RANK
        + offs_q_d_lora_pv[None, :]
    )
    gl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=query_mask_0_pv[:, None] & query_mask_1_pv[:, None],
    )
    segm_offset = (
        query_offset_0_qk.to(gl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1_qk * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    gl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0_qk & query_mask_1_qk)
    gl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0_qk & query_mask_1_qk)


@triton.jit
def _mla_decode_fwd_reduce_kernel(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    out_scale_ptr,  # float32
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_tables_stride: tl.int64,  # int
    num_tokens_per_seq: tl.int32,
    TILE_SIZE: tl.constexpr,  # int
    KV_LORA_RANK: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    ALL_DECODE: tl.constexpr = False,  # int
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    out_scale = None
    if out_scale_ptr is not None:
        out_scale = 1 / tl.load(out_scale_ptr)

    if ALL_DECODE:
        seq_idx = query_token_idx
    else:
        seq_idx = query_token_idx // num_tokens_per_seq

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.math.exp2(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * KV_LORA_RANK
        + tl.arange(0, KV_LORA_RANK)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None],
        other=0.0,
    )
    segm_output *= tl.math.exp2(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if out_scale_ptr is not None:
        acc = acc * out_scale

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, KV_LORA_RANK)
    )
    tl.store(output_ptr + output_offset, acc)
