# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import torch
import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.common_utils import strip_annotate
from aiter.ops.triton.utils.types import e4m3_dtype

float8_info = torch.finfo(e4m3_dtype)


@triton.jit
def fast_exp(x):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    return tl.math.exp2(x * RCP_LN2)


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.math.exp2(Sdiv)
    p2 = tl.math.exp2(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


_kernel_unified_attention_2d_repr = make_kernel_repr(
    "kernel_unified_attention_2d",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "TILE_SIZE",
        "HEAD_SIZE",
        "HEAD_SIZE_PADDED",
        "V_HEAD_SIZE",
        "V_HEAD_SIZE_PADDED",
        "USE_ALIBI_SLOPES",
        "USE_QQ_BIAS",
        "USE_SOFTCAP",
        "USE_SINKS",
        "SLIDING_WINDOW",
        "BLOCK_Q",
        "BLOCK_M",
        "ALL_DECODE",
        "SHUFFLED_KV_CACHE",
        "SPLIT_UNMASKED_LOOP",
        "K_WIDTH",
    ],
)


kernel_unified_attention_3d_repr = make_kernel_repr(
    "kernel_unified_attention_3d",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "TILE_SIZE",
        "HEAD_SIZE",
        "V_HEAD_SIZE",
        "NUM_SEGMENTS_PER_SEQ",
        "num_warps",
        "waves_per_eu",
        "num_stages",
        "ALL_DECODE",
        "SHUFFLED_KV_CACHE",
        "IS_Q_FP8",
        "IS_KV_FP8",
    ],
)


def _kernel_unified_attention_repr(specialization):
    # keep the 2d / 3d kernel names
    if specialization.constants.get("GRID_3D", False):
        return kernel_unified_attention_3d_repr(specialization)
    return _kernel_unified_attention_2d_repr(specialization)


@aggregate
@strip_annotate
class AttentionConfig:
    """Compile-time sizes and feature flags of the unified attention kernel."""

    NUM_QUERY_HEADS: tl.constexpr
    NUM_QUERIES_PER_KV: tl.constexpr
    BLOCK_SIZE: tl.constexpr
    TILE_SIZE: tl.constexpr
    HEAD_SIZE_PADDED: tl.constexpr
    V_HEAD_SIZE_PADDED: tl.constexpr
    BLOCK_Q: tl.constexpr
    BLOCK_M: tl.constexpr
    USE_ALIBI_SLOPES: tl.constexpr
    USE_QQ_BIAS: tl.constexpr
    USE_SOFTCAP: tl.constexpr
    USE_SINKS: tl.constexpr
    SLIDING_WINDOW: tl.constexpr
    ALL_DECODE: tl.constexpr
    SHUFFLED_KV_CACHE: tl.constexpr
    SPLIT_UNMASKED_LOOP: tl.constexpr
    K_WIDTH: tl.constexpr
    SLOT_SIZE: tl.constexpr
    NUM_SLOTS: tl.constexpr
    NUM_SEGMENTS_PER_SEQ: tl.constexpr
    RCP_LN2: tl.constexpr
    KV_CACHE_MODIFIER: tl.constexpr
    stride_k_cache_3: tl.constexpr
    stride_v_cache_3: tl.constexpr

    @triton.constexpr_function
    def __init__(
        self,
        NUM_QUERY_HEADS,
        NUM_QUERIES_PER_KV,
        BLOCK_SIZE,
        TILE_SIZE,
        HEAD_SIZE_PADDED,
        V_HEAD_SIZE_PADDED,
        BLOCK_Q,
        BLOCK_M,
        USE_ALIBI_SLOPES,
        USE_QQ_BIAS,
        USE_SOFTCAP,
        USE_SINKS,
        SLIDING_WINDOW,
        ALL_DECODE,
        SHUFFLED_KV_CACHE,
        SPLIT_UNMASKED_LOOP,
        K_WIDTH,
        NUM_SEGMENTS_PER_SEQ,
        stride_k_cache_3,
        stride_v_cache_3,
    ):
        self.NUM_QUERY_HEADS = tl.constexpr(NUM_QUERY_HEADS)
        self.NUM_QUERIES_PER_KV = tl.constexpr(NUM_QUERIES_PER_KV)
        self.BLOCK_SIZE = tl.constexpr(BLOCK_SIZE)
        self.TILE_SIZE = tl.constexpr(TILE_SIZE)
        self.HEAD_SIZE_PADDED = tl.constexpr(HEAD_SIZE_PADDED)
        self.V_HEAD_SIZE_PADDED = tl.constexpr(V_HEAD_SIZE_PADDED)
        self.BLOCK_Q = tl.constexpr(BLOCK_Q)
        self.BLOCK_M = tl.constexpr(BLOCK_M)
        self.USE_ALIBI_SLOPES = tl.constexpr(USE_ALIBI_SLOPES)
        self.USE_QQ_BIAS = tl.constexpr(USE_QQ_BIAS)
        self.USE_SOFTCAP = tl.constexpr(USE_SOFTCAP)
        self.USE_SINKS = tl.constexpr(USE_SINKS)
        self.SLIDING_WINDOW = tl.constexpr(SLIDING_WINDOW)
        self.ALL_DECODE = tl.constexpr(ALL_DECODE)
        self.SHUFFLED_KV_CACHE = tl.constexpr(SHUFFLED_KV_CACHE)
        self.SPLIT_UNMASKED_LOOP = tl.constexpr(SPLIT_UNMASKED_LOOP)
        self.K_WIDTH = tl.constexpr(K_WIDTH)
        # a shuffled tile is NUM_SLOTS slots of SLOT_SIZE keys, each within one page
        self.SLOT_SIZE = tl.constexpr(min(TILE_SIZE, BLOCK_SIZE))
        self.NUM_SLOTS = tl.constexpr(TILE_SIZE // min(TILE_SIZE, BLOCK_SIZE))
        self.NUM_SEGMENTS_PER_SEQ = tl.constexpr(NUM_SEGMENTS_PER_SEQ)
        # needed to use exp2 (exp2 -> exp conversion)
        self.RCP_LN2 = tl.constexpr(1.4426950408889634)
        self.KV_CACHE_MODIFIER = tl.constexpr(".cg" if ALL_DECODE else "")
        self.stride_k_cache_3 = tl.constexpr(stride_k_cache_3)
        self.stride_v_cache_3 = tl.constexpr(stride_v_cache_3)


@triton.constexpr_function
def _as_field(x):
    # __init__ gets compile-time values (int args of 1, None, folded scales) unwrapped
    return x if isinstance(x, tl.tensor) else tl.constexpr(x)


@aggregate
@strip_annotate
class KVLoader:
    """Loads K/V tiles of one sequence from the paged cache.

    Plain: K/V = [num_blks, blk_size, num_kv_heads, head_size].
    Shuffled: K = [num_blks, num_kv_heads, head_size // W, blk_size, W],
    V = [num_blks, num_kv_heads, blk_size // W, head_size, W], read a page at a
    time and un-shuffled in registers.

    Offsets and masks are built before either load, so the loads can be reordered.
    """

    cfg: AttentionConfig

    key_cache_ptr: tl.tensor
    value_cache_ptr: tl.tensor
    block_tables_ptr: tl.tensor
    block_table_offset: tl.tensor  # this sequence's row
    kv_head_idx: tl.tensor
    stride_k_cache_0: tl.tensor | tl.constexpr
    stride_k_cache_1: tl.tensor | tl.constexpr
    stride_k_cache_2: tl.tensor | tl.constexpr
    stride_v_cache_0: tl.tensor | tl.constexpr
    stride_v_cache_1: tl.tensor | tl.constexpr
    stride_v_cache_2: tl.tensor | tl.constexpr
    offs_t: tl.tensor
    offs_d: tl.tensor
    offs_vd: tl.tensor
    offs_shfl: tl.tensor | tl.constexpr
    dim_mask: tl.tensor
    v_dim_mask: tl.tensor
    max_seq_prefix_len: tl.tensor

    @triton.constexpr_function
    def __init__(
        self,
        cfg,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        block_table_offset,
        kv_head_idx,
        stride_k_cache_0,
        stride_k_cache_1,
        stride_k_cache_2,
        stride_v_cache_0,
        stride_v_cache_1,
        stride_v_cache_2,
        offs_t,
        offs_d,
        offs_vd,
        offs_shfl,
        dim_mask,
        v_dim_mask,
        max_seq_prefix_len,
    ):
        self.cfg = cfg
        self.key_cache_ptr = key_cache_ptr
        self.value_cache_ptr = value_cache_ptr
        self.block_tables_ptr = block_tables_ptr
        self.block_table_offset = block_table_offset
        self.kv_head_idx = kv_head_idx
        self.stride_k_cache_0 = _as_field(stride_k_cache_0)
        self.stride_k_cache_1 = _as_field(stride_k_cache_1)
        self.stride_k_cache_2 = _as_field(stride_k_cache_2)
        self.stride_v_cache_0 = _as_field(stride_v_cache_0)
        self.stride_v_cache_1 = _as_field(stride_v_cache_1)
        self.stride_v_cache_2 = _as_field(stride_v_cache_2)
        self.offs_t = offs_t
        self.offs_d = offs_d
        self.offs_vd = offs_vd
        self.offs_shfl = _as_field(offs_shfl)
        self.dim_mask = dim_mask
        self.v_dim_mask = v_dim_mask
        self.max_seq_prefix_len = max_seq_prefix_len

    @triton.jit
    def load_tile(self, j, target_dtype: tl.constexpr, MASKED: tl.constexpr):
        """Returns K (HEAD_SIZE_PADDED, TILE_SIZE), V (TILE_SIZE, V_HEAD_SIZE_PADDED)
        and the tile's key positions. Without MASKED every key of the tile is
        assumed to be in range."""
        seq_offset = j * self.cfg.TILE_SIZE + self.offs_t
        tile_mask = self.tile_mask(seq_offset, MASKED)
        block_idx = self.block_ids(j, seq_offset)
        # address math before either load; moving it changes the schedule
        if self.cfg.SHUFFLED_KV_CACHE:
            k_offset = self.k_offset(j, seq_offset, block_idx)
            v_offset = self.v_offset(j, seq_offset, block_idx)
            k_mask = self.k_mask(j, tile_mask)
            v_mask = self.v_mask(j, tile_mask)
        else:
            v_offset = self.v_offset(j, seq_offset, block_idx)
            v_mask = self.v_mask(j, tile_mask)
            k_offset = self.k_offset(j, seq_offset, block_idx)
            k_mask = self.k_mask(j, tile_mask)
        K = self.load_k(k_offset, k_mask, target_dtype)
        V = self.load_v(v_offset, v_mask, target_dtype)
        return K, V, seq_offset

    @triton.jit
    def tile_mask(self, seq_offset, MASKED: tl.constexpr):
        # None when every key is in range
        mask = None
        if MASKED and not self.cfg.SHUFFLED_KV_CACHE:
            # to reduce the masking effect when not needed
            if self.cfg.TILE_SIZE == self.cfg.BLOCK_SIZE:
                mask = tl.full((1,), 1, dtype=tl.int1)
            else:
                mask = seq_offset < self.max_seq_prefix_len
        return mask

    @triton.jit
    def slot_start(self, j):
        cfg = self.cfg
        return j * cfg.TILE_SIZE + tl.arange(0, cfg.NUM_SLOTS) * cfg.SLOT_SIZE

    @triton.jit
    def slot_mask(self, j):
        # also keeps the block table read inside this sequence
        return self.slot_start(j) < self.max_seq_prefix_len

    @triton.jit
    def block_ids(self, j, seq_offset):
        """Physical page of every key (plain), of the tile (one-page shuffled) or
        of every slot (other shuffled tiles)."""
        cfg = self.cfg
        row_ptr = self.block_tables_ptr + self.block_table_offset
        if not cfg.SHUFFLED_KV_CACHE:
            block_idx = tl.load(row_ptr + seq_offset // cfg.BLOCK_SIZE)
        elif cfg.TILE_SIZE == cfg.BLOCK_SIZE:
            block_idx = tl.load(row_ptr + j)
        elif cfg.NUM_SLOTS > 1:
            block_idx = tl.load(
                row_ptr + self.slot_start(j) // cfg.BLOCK_SIZE,
                mask=self.slot_mask(j),
                other=0,
            )
        else:
            block_idx = tl.load(row_ptr + self.slot_start(j) // cfg.BLOCK_SIZE)
        return block_idx.to(tl.int64)

    @triton.jit
    def k_offset(self, j, seq_offset, block_idx):
        cfg = self.cfg
        if not cfg.SHUFFLED_KV_CACHE:
            offset = (
                block_idx[None, :] * self.stride_k_cache_0
                + self.kv_head_idx * self.stride_k_cache_2
                + self.offs_d[:, None] * cfg.stride_k_cache_3
                + (seq_offset % cfg.BLOCK_SIZE)[None, :] * self.stride_k_cache_1
            )
        elif cfg.TILE_SIZE == cfg.BLOCK_SIZE:
            offset = (
                block_idx * self.stride_k_cache_0
                + self.kv_head_idx * self.stride_k_cache_1
                + self.offs_shfl
            )
        elif cfg.NUM_SLOTS > 1:
            # whole pages, flattened in the one-page [HEAD_SIZE // W, TILE, W] order
            W: tl.constexpr = cfg.K_WIDTH
            offset = (
                (
                    block_idx * self.stride_k_cache_0
                    + self.kv_head_idx * self.stride_k_cache_1
                )[None, :, None]
                + tl.arange(0, cfg.HEAD_SIZE_PADDED // W)[:, None, None]
                * (cfg.BLOCK_SIZE * W)
                + tl.arange(0, cfg.BLOCK_SIZE * W)[None, None, :]
            ).reshape(cfg.TILE_SIZE * cfg.HEAD_SIZE_PADDED)
        else:
            # part of one page
            W: tl.constexpr = cfg.K_WIDTH
            in_page = self.slot_start(j) % cfg.BLOCK_SIZE
            offset = (
                block_idx * self.stride_k_cache_0
                + self.kv_head_idx * self.stride_k_cache_1
                + in_page * W
            )[:, None, None] + (
                tl.arange(0, cfg.HEAD_SIZE_PADDED // W)[None, :, None]
                * (cfg.BLOCK_SIZE * W)
                + tl.arange(0, cfg.SLOT_SIZE * W)[None, None, :]
            )
        return offset

    @triton.jit
    def v_offset(self, j, seq_offset, block_idx):
        cfg = self.cfg
        if not cfg.SHUFFLED_KV_CACHE:
            offset = (
                block_idx[:, None] * self.stride_v_cache_0
                + self.kv_head_idx * self.stride_v_cache_2
                + self.offs_vd[None, :] * cfg.stride_v_cache_3
                + (seq_offset % cfg.BLOCK_SIZE)[:, None] * self.stride_v_cache_1
            )
        elif cfg.TILE_SIZE == cfg.BLOCK_SIZE:
            offset = (
                block_idx * self.stride_v_cache_0
                + self.kv_head_idx * self.stride_v_cache_1
                + self.offs_shfl
            )
        elif cfg.NUM_SLOTS > 1:
            # whole pages; concatenated they are already the one-page order
            offset = (
                (
                    block_idx * self.stride_v_cache_0
                    + self.kv_head_idx * self.stride_v_cache_1
                )[:, None]
                + tl.arange(0, cfg.BLOCK_SIZE * cfg.HEAD_SIZE_PADDED)[None, :]
            ).reshape(cfg.TILE_SIZE * cfg.HEAD_SIZE_PADDED)
        else:
            # part of one page: one contiguous run
            in_page = self.slot_start(j) % cfg.BLOCK_SIZE
            offset = (
                block_idx * self.stride_v_cache_0
                + self.kv_head_idx * self.stride_v_cache_1
                + (in_page // cfg.K_WIDTH) * self.stride_v_cache_2
            )[:, None] + tl.arange(0, cfg.SLOT_SIZE * cfg.HEAD_SIZE_PADDED)[None, :]
        return offset

    @triton.jit
    def k_mask(self, j, tile_mask):
        # None when nothing needs masking
        cfg = self.cfg
        mask = None
        if not cfg.SHUFFLED_KV_CACHE:
            if tile_mask is None:
                mask = self.dim_mask[:, None]
            else:
                mask = self.dim_mask[:, None] & tile_mask[None, :]
        elif cfg.NUM_SLOTS > 1:
            mask = tl.broadcast_to(
                self.slot_mask(j)[None, :, None],
                (
                    cfg.HEAD_SIZE_PADDED // cfg.K_WIDTH,
                    cfg.NUM_SLOTS,
                    cfg.BLOCK_SIZE * cfg.K_WIDTH,
                ),
            ).reshape(cfg.TILE_SIZE * cfg.HEAD_SIZE_PADDED)
        return mask

    @triton.jit
    def v_mask(self, j, tile_mask):
        # None when nothing needs masking
        cfg = self.cfg
        mask = None
        if not cfg.SHUFFLED_KV_CACHE:
            if tile_mask is None:
                mask = self.v_dim_mask[None, :]
            else:
                mask = self.v_dim_mask[None, :] & tile_mask[:, None]
        elif cfg.NUM_SLOTS > 1:
            mask = tl.broadcast_to(
                self.slot_mask(j)[:, None],
                (cfg.NUM_SLOTS, cfg.BLOCK_SIZE * cfg.HEAD_SIZE_PADDED),
            ).reshape(cfg.TILE_SIZE * cfg.HEAD_SIZE_PADDED)
        return mask

    @triton.jit
    def load_k(self, k_offset, k_mask, target_dtype: tl.constexpr):
        """K : (HEAD_SIZE_PADDED, TILE_SIZE)"""
        cfg = self.cfg
        ptrs = self.key_cache_ptr + k_offset
        if k_mask is None:
            K = tl.load(ptrs, cache_modifier=cfg.KV_CACHE_MODIFIER)
        else:
            K = tl.load(
                ptrs, mask=k_mask, other=0.0, cache_modifier=cfg.KV_CACHE_MODIFIER
            )
        K = K.to(target_dtype)

        if cfg.SHUFFLED_KV_CACHE:
            K = (
                K.reshape(
                    cfg.HEAD_SIZE_PADDED // cfg.K_WIDTH,
                    cfg.TILE_SIZE,
                    cfg.K_WIDTH,
                )
                .permute(1, 0, 2)
                .reshape(cfg.TILE_SIZE, cfg.HEAD_SIZE_PADDED)
                .trans(1, 0)
            )
        return K

    @triton.jit
    def load_v(self, v_offset, v_mask, target_dtype: tl.constexpr):
        """V : (TILE_SIZE, V_HEAD_SIZE_PADDED)"""
        cfg = self.cfg
        ptrs = self.value_cache_ptr + v_offset
        if v_mask is None:
            V = tl.load(ptrs, cache_modifier=cfg.KV_CACHE_MODIFIER)
        else:
            V = tl.load(
                ptrs, mask=v_mask, other=0.0, cache_modifier=cfg.KV_CACHE_MODIFIER
            )
        V = V.to(target_dtype)

        if cfg.SHUFFLED_KV_CACHE:
            V = (
                V.reshape(
                    cfg.TILE_SIZE // cfg.K_WIDTH,
                    cfg.HEAD_SIZE_PADDED,
                    cfg.K_WIDTH,
                )
                .permute(0, 2, 1)
                .reshape(cfg.TILE_SIZE, cfg.HEAD_SIZE_PADDED)
            )
        return V


@aggregate
@strip_annotate
class AttentionProgram:
    """Per-program state: query block, masks, scales and the KV tile range
    [tile_start, tile_end); unmasked_tile_end is only set with SPLIT_UNMASKED_LOOP."""

    cfg: AttentionConfig

    q: tl.tensor
    query_pos: tl.tensor
    query_offset_0: tl.tensor
    query_offset_1: tl.tensor
    query_mask_0: tl.tensor
    query_mask_1: tl.tensor
    offs_vd: tl.tensor
    v_dim_mask: tl.tensor
    context_len: tl.tensor
    max_seq_prefix_len: tl.tensor
    tile_start: tl.tensor | tl.constexpr
    tile_end: tl.tensor
    unmasked_tile_end: tl.tensor | tl.constexpr
    qk_scale: tl.tensor | tl.constexpr
    softcap: tl.tensor | tl.constexpr
    v_descale: tl.tensor | tl.constexpr
    alibi_slope: tl.tensor | tl.constexpr
    qq_bias_row_ptrs: tl.tensor | tl.constexpr
    qq_bias_stride_0: tl.tensor | tl.constexpr

    @triton.constexpr_function
    def __init__(
        self,
        cfg,
        q,
        query_pos,
        query_offset_0,
        query_offset_1,
        query_mask_0,
        query_mask_1,
        offs_vd,
        v_dim_mask,
        context_len,
        max_seq_prefix_len,
        tile_start,
        tile_end,
        unmasked_tile_end,
        qk_scale,
        softcap,
        v_descale,
        alibi_slope,
        qq_bias_row_ptrs,
        qq_bias_stride_0,
    ):
        self.cfg = cfg
        self.q = q
        self.query_pos = query_pos
        self.query_offset_0 = query_offset_0
        self.query_offset_1 = query_offset_1
        self.query_mask_0 = query_mask_0
        self.query_mask_1 = query_mask_1
        self.offs_vd = offs_vd
        self.v_dim_mask = v_dim_mask
        self.context_len = context_len
        self.max_seq_prefix_len = max_seq_prefix_len
        self.tile_start = _as_field(tile_start)
        self.tile_end = tile_end
        self.unmasked_tile_end = _as_field(unmasked_tile_end)
        self.qk_scale = _as_field(qk_scale)
        self.softcap = _as_field(softcap)
        self.v_descale = _as_field(v_descale)
        self.alibi_slope = _as_field(alibi_slope)
        self.qq_bias_row_ptrs = _as_field(qq_bias_row_ptrs)
        self.qq_bias_stride_0 = _as_field(qq_bias_stride_0)

    @triton.jit
    def initialize(
        cfg,
        q,
        query_pos,
        query_offset_0,
        query_offset_1,
        query_mask_0,
        query_mask_1,
        offs_vd,
        v_dim_mask,
        seq_len,
        q_block_local_idx,
        cur_batch_query_len,
        segm_idx,
        tiles_per_segment,
        scale: tl.constexpr,
        softcap,
        q_descale_ptr,
        k_descale_ptr,
        v_descale_ptr,
        alibi_slopes_ptr,
        qq_bias_ptr,
        qq_bias_stride_0,
    ):
        # context length for this particular sequences
        context_len = seq_len - cur_batch_query_len

        # alibi slope for this head
        alibi_slope = None
        if cfg.USE_ALIBI_SLOPES:
            alibi_slope = tl.load(
                alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
            )

        # query-query attention bias
        qq_bias_row_ptrs = None
        if cfg.USE_QQ_BIAS:
            qq_bias_row_ptrs = (
                qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
            )  # shape: [BLOCK_M]

        # compute the length of the longest sequence prefix spanned by any
        # query token in the current q_block (q_block_local_idx)
        max_seq_prefix_len = (
            context_len
            + q_block_local_idx * cfg.BLOCK_Q
            + (cfg.BLOCK_M - 1) // cfg.NUM_QUERIES_PER_KV
            + 1
        )

        # adjust for potential padding in the last q_block by considering the
        # actual sequence length
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

        # calculate the number of tiles that need to be processed to
        # cover the longest sequence prefix (due to causal masking, tiles beyond
        # this prefix can be skipped)
        num_tiles = cdiv_fn(max_seq_prefix_len, cfg.TILE_SIZE)

        # sliding window: skip the tiles outside the window
        tile_start = 0
        tile_end = num_tiles
        if cfg.SLIDING_WINDOW > 0:
            # Query rows covered by this Q-block
            qpos_lo = q_block_local_idx * cfg.BLOCK_Q
            qpos_hi = tl.minimum(
                qpos_lo + (cfg.BLOCK_M - 1) // cfg.NUM_QUERIES_PER_KV,
                cur_batch_query_len - 1,
            )
            # For sliding window, each query position q can only attend to
            # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
            # where q_abs = context_len + q
            # The union of allowed key positions for this Q-block is:
            # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
            first_allowed_key = context_len + qpos_lo - cfg.SLIDING_WINDOW + 1
            last_allowed_key = context_len + qpos_hi
            # Convert to tile indices and clamp
            tile_start = tl.maximum(0, first_allowed_key // cfg.TILE_SIZE)
            tile_end = tl.minimum((last_allowed_key // cfg.TILE_SIZE) + 1, num_tiles)

        # split-KV: reduce_segments rebuilds the same partition from seq_len
        if cfg.NUM_SEGMENTS_PER_SEQ > 1:
            segm_tile_start = segm_idx * tiles_per_segment
            if cfg.SLIDING_WINDOW > 0:
                tile_start = tl.maximum(tile_start, segm_tile_start)
            else:
                tile_start = segm_tile_start
            tile_end = tl.minimum((segm_idx + 1) * tiles_per_segment, tile_end)

        # qk_scale = scale * RCP_LN2 (log_2 e) so that we can use exp2 later;
        # a local float keeps the product in fp32
        RCP_LN2 = 1.4426950408889634
        qk_scale = scale * RCP_LN2
        if q_descale_ptr is not None:
            qk_scale = qk_scale * tl.load(q_descale_ptr)
        k_descale = None
        v_descale = None
        if k_descale_ptr is not None:
            k_descale = tl.load(k_descale_ptr)
        if v_descale_ptr is not None:
            v_descale = tl.load(v_descale_ptr)
        if k_descale is not None:
            qk_scale = qk_scale * k_descale

        unmasked_tile_end = tile_start
        if cfg.SPLIT_UNMASKED_LOOP:
            # every query of the block sees the whole tile below this key
            min_query_key_limit = context_len + q_block_local_idx * cfg.BLOCK_Q + 1
            unmasked_tile_end = tl.minimum(
                min_query_key_limit // cfg.TILE_SIZE, tile_end
            )
            unmasked_tile_end = tl.maximum(unmasked_tile_end, tile_start)

        return AttentionProgram(
            cfg,
            q,
            query_pos,
            query_offset_0,
            query_offset_1,
            query_mask_0,
            query_mask_1,
            offs_vd,
            v_dim_mask,
            context_len,
            max_seq_prefix_len,
            tile_start,
            tile_end,
            unmasked_tile_end,
            qk_scale,
            softcap,
            v_descale,
            alibi_slope,
            qq_bias_row_ptrs,
            qq_bias_stride_0,
        )

    @triton.jit
    def compute_qk(self, K):
        # S : (BLOCK_M, TILE_SIZE)
        S = self.qk_scale * tl.dot(self.q, K)

        if self.cfg.USE_SOFTCAP:
            # softcap here uses exp2 and consumes RCP_LN2 conversion.
            # multiply by RCP_LN2 again to be used in later exp2
            S = apply_softcap(S, self.softcap) * self.cfg.RCP_LN2
        return S

    @triton.jit
    def causal_mask(self, seq_offset):
        return seq_offset[None, :] < self.context_len + self.query_pos[:, None] + 1

    @triton.jit
    def apply_mask_qk(self, S, seq_offset, seq_mask):
        S = tl.where(
            self.query_mask_1[:, None] & self.query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )

        if self.cfg.SLIDING_WINDOW > 0:
            S = tl.where(
                (self.context_len + self.query_pos[:, None] - seq_offset)
                < self.cfg.SLIDING_WINDOW,
                S,
                float("-inf"),
            )
        return S

    @triton.jit
    def apply_score_bias(self, S, seq_offset):
        if self.cfg.USE_ALIBI_SLOPES:
            # prescale w. RCP_LN2 for later exp2
            S += (
                self.alibi_slope[:, None]
                * (seq_offset - self.context_len)
                * self.cfg.RCP_LN2
            )

        if self.cfg.USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - self.context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < self.qq_bias_stride_0
            qq_bias = tl.load(
                self.qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            # prescale w. RCP_LN2 for later exp2
            S += qq_bias * self.cfg.RCP_LN2
        return S

    @triton.jit
    def softmax_update(self, S, M, L, acc):
        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j
        return P, M, L, acc

    @triton.jit
    def compute_pv(self, P, V, acc):
        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        return tl.dot(P.to(V.dtype), V, acc=acc)

    @triton.jit
    def store_output(
        self,
        acc,
        L,
        output_ptr,
        output_stride_0,
        output_stride_1,
        out_scale_ptr,
        FP8_MIN: tl.constexpr,
        FP8_MAX: tl.constexpr,
    ):
        # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
        if self.v_descale is not None:
            one_over_L = self.v_descale / L[:, None]
        else:
            one_over_L = 1.0 / L[:, None]
        acc = acc * one_over_L
        if out_scale_ptr is not None:
            acc = acc / tl.load(out_scale_ptr)

        if output_ptr.type.element_ty.is_fp8():
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

        output_offset = (
            self.query_offset_0[:, None] * output_stride_0
            + self.query_offset_1[:, None] * output_stride_1
            + self.offs_vd[None, :]
        )

        tl.store(
            output_ptr + output_offset,
            acc,
            mask=self.v_dim_mask[None, :]
            & self.query_mask_0[:, None]
            & self.query_mask_1[:, None],
        )

    @triton.jit
    def store_partial(
        self,
        acc,
        M,
        L,
        segm_output_ptr,
        segm_max_ptr,
        segm_expsum_ptr,
        segm_idx,
    ):
        """Split-KV partials for reduce_segments, which also applies out_scale:
        segm_output: [num_tokens, num_query_heads, NUM_SEGMENTS_PER_SEQ, V_HEAD_SIZE_PADDED]
        segm_max / segm_expsum: [num_tokens, num_query_heads, NUM_SEGMENTS_PER_SEQ]
        """
        cfg = self.cfg
        if self.v_descale is not None:
            acc = acc * self.v_descale

        segm_output_offset = (
            self.query_offset_0[:, None].to(tl.int64)
            * (cfg.NUM_QUERY_HEADS * cfg.NUM_SEGMENTS_PER_SEQ * cfg.V_HEAD_SIZE_PADDED)
            + self.query_offset_1[:, None]
            * (cfg.NUM_SEGMENTS_PER_SEQ * cfg.V_HEAD_SIZE_PADDED)
            + segm_idx * cfg.V_HEAD_SIZE_PADDED
            + self.offs_vd[None, :]
        )
        tl.store(
            segm_output_ptr + segm_output_offset,
            acc,
            mask=self.v_dim_mask[None, :]
            & self.query_mask_0[:, None]
            & self.query_mask_1[:, None],
        )
        segm_offset = (
            self.query_offset_0.to(tl.int64)
            * (cfg.NUM_QUERY_HEADS * cfg.NUM_SEGMENTS_PER_SEQ)
            + self.query_offset_1 * cfg.NUM_SEGMENTS_PER_SEQ
            + segm_idx
        )
        row_mask = self.query_mask_0 & self.query_mask_1
        tl.store(segm_max_ptr + segm_offset, M, mask=row_mask)
        tl.store(segm_expsum_ptr + segm_offset, L, mask=row_mask)


@triton.jit
def initial_row_max(cfg, sink_ptr, query_offset_1, query_mask_1, segm_idx):
    M = tl.full([cfg.BLOCK_M], float("-inf"), dtype=tl.float32)
    if cfg.USE_SINKS:
        # only the first segment counts the sink
        if cfg.NUM_SEGMENTS_PER_SEQ == 1 or segm_idx == 0:
            # Prescale with RCP_LN2, needed for exp2
            M = (
                tl.load(
                    sink_ptr + query_offset_1,
                    mask=query_mask_1,
                    other=float("-inf"),
                ).to(dtype=tl.float32)
                * cfg.RCP_LN2
            )
    return M


@triton.jit
def attention_loop(
    pgm, kv_loader, M, L, acc, tile_start, tile_end, MASKED: tl.constexpr
):
    """Online softmax over tiles [tile_start, tile_end)."""
    for j in range(tile_start, tile_end):
        K, V, seq_offset = kv_loader.load_tile(j, pgm.q.dtype, MASKED)
        # position-only, so it can overlap the QK dot
        if MASKED:
            seq_mask = pgm.causal_mask(seq_offset)
        S = pgm.compute_qk(K)
        if MASKED:
            S = pgm.apply_mask_qk(S, seq_offset, seq_mask)
        S = pgm.apply_score_bias(S, seq_offset)
        P, M, L, acc = pgm.softmax_update(S, M, L, acc)
        acc = pgm.compute_pv(P, V, acc)
    return M, L, acc


@triton.jit(repr=_kernel_unified_attention_repr)
def kernel_unified_attention(
    output_ptr,  # [num_tokens, num_query_heads, head_size], None when split
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size], None unless split
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale: tl.constexpr,  # float32
    q_descale_ptr,  # float32
    k_descale_ptr,  # float32
    v_descale_ptr,  # float32
    out_scale_ptr,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    V_HEAD_SIZE: tl.constexpr,  # int, value head size (may differ from HEAD_SIZE)
    V_HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    ALL_DECODE: tl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: tl.constexpr = False,  # bool
    SPLIT_UNMASKED_LOOP: tl.constexpr = False,  # bool
    K_WIDTH: tl.constexpr = 0,  # int
    # Split-KV (3d grid)
    GRID_3D: tl.constexpr = False,  # bool, (q_block, kv_head, segment) grid
    NUM_SEGMENTS_PER_SEQ: tl.constexpr = 1,  # int
    # only used to name the 3d kernel
    num_warps: tl.constexpr = 4,  # int
    waves_per_eu: tl.constexpr = 0,  # int
    num_stages: tl.constexpr = 2,  # int
    IS_Q_FP8: tl.constexpr = False,  # bool
    IS_KV_FP8: tl.constexpr = False,  # bool
):
    """Paged causal attention, one program per (query block, kv head[, segment]).

    GRID_3D=False: grid (num_kv_heads, num_q_blocks).
    GRID_3D=True: grid (num_q_blocks, num_kv_heads, NUM_SEGMENTS_PER_SEQ); with more
    than one segment the program writes fp32 partials for reduce_segments.
    """
    # SPLIT_UNMASKED_LOOP does not support SHUFFLED_KV_CACHE or SLIDING_WINDOW.
    tl.static_assert(
        not (SPLIT_UNMASKED_LOOP and SHUFFLED_KV_CACHE),
        "SPLIT_UNMASKED_LOOP does not support SHUFFLED_KV_CACHE",
    )
    tl.static_assert(
        not (SPLIT_UNMASKED_LOOP and SLIDING_WINDOW > 0),
        "SPLIT_UNMASKED_LOOP does not support sliding-window attention",
    )
    tl.static_assert(
        GRID_3D or NUM_SEGMENTS_PER_SEQ == 1,
        "NUM_SEGMENTS_PER_SEQ > 1 needs the 3d grid",
    )
    tl.static_assert(
        not SHUFFLED_KV_CACHE
        or (TILE_SIZE % K_WIDTH == 0 and BLOCK_SIZE % K_WIDTH == 0),
        "SHUFFLED_KV_CACHE needs TILE_SIZE and BLOCK_SIZE to be multiples of K_WIDTH",
    )

    if GRID_3D:
        q_block_global_idx = tl.program_id(0)
        kv_head_idx = tl.program_id(1)
        segm_idx = tl.program_id(2)
    else:
        kv_head_idx = tl.program_id(0)
        q_block_global_idx = tl.program_id(1)
        segm_idx = 0

    cfg = AttentionConfig(
        num_query_heads,
        num_queries_per_kv,
        BLOCK_SIZE,
        TILE_SIZE,
        HEAD_SIZE_PADDED,
        V_HEAD_SIZE_PADDED,
        BLOCK_Q,
        BLOCK_M,
        USE_ALIBI_SLOPES,
        USE_QQ_BIAS,
        USE_SOFTCAP,
        USE_SINKS,
        SLIDING_WINDOW,
        ALL_DECODE,
        SHUFFLED_KV_CACHE,
        SPLIT_UNMASKED_LOOP,
        K_WIDTH,
        NUM_SEGMENTS_PER_SEQ,
        stride_k_cache_3,
        stride_v_cache_3,
    )

    if ALL_DECODE:
        seq_idx = q_block_global_idx
        q_block_local_idx: tl.int32 = 0
        cur_batch_query_len: tl.int32 = 1
        cur_batch_in_all_start_index: tl.int32 = q_block_global_idx
    else:
        seq_idx = find_seq_idx(
            query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
        )

        q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

        q_block_local_idx = q_block_global_idx - q_block_start_idx

        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

        if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
            return

    tiles_per_segment = 0
    if NUM_SEGMENTS_PER_SEQ > 1:
        # sequence len for this particular sequence
        seq_len = tl.load(seq_lens_ptr + seq_idx)
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        # This segment owns no tiles
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_vd = tl.arange(0, V_HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    offs_shfl = None
    if SHUFFLED_KV_CACHE:
        offs_shfl = tl.arange(0, TILE_SIZE * HEAD_SIZE_PADDED)

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    if V_HEAD_SIZE_PADDED != V_HEAD_SIZE:
        v_dim_mask = offs_vd < V_HEAD_SIZE
    else:
        v_dim_mask = tl.full((1,), 1, dtype=tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # the 3d launch keeps Q cached for the other segments of the query block
    if not GRID_3D and (ALL_DECODE or BLOCK_M >= num_query_heads):
        Q_cache_modifier: tl.constexpr = ".cg"
    else:
        Q_cache_modifier: tl.constexpr = ""
    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
        cache_modifier=Q_cache_modifier,
    )

    block_table_offset = seq_idx * block_table_stride

    M = initial_row_max(cfg, sink_ptr, query_offset_1, query_mask_1, segm_idx)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_SIZE_PADDED], dtype=tl.float32)

    if NUM_SEGMENTS_PER_SEQ == 1:
        # sequence len for this particular sequence
        seq_len = tl.load(seq_lens_ptr + seq_idx)

    pgm = AttentionProgram.initialize(
        cfg,
        Q,
        query_pos,
        query_offset_0,
        query_offset_1,
        query_mask_0,
        query_mask_1,
        offs_vd,
        v_dim_mask,
        seq_len,
        q_block_local_idx,
        cur_batch_query_len,
        segm_idx,
        tiles_per_segment,
        scale,
        softcap,
        q_descale_ptr,
        k_descale_ptr,
        v_descale_ptr,
        alibi_slopes_ptr,
        qq_bias_ptr,
        qq_bias_stride_0,
    )

    kv_loader = KVLoader(
        cfg,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        block_table_offset,
        kv_head_idx,
        stride_k_cache_0,
        stride_k_cache_1,
        stride_k_cache_2,
        stride_v_cache_0,
        stride_v_cache_1,
        stride_v_cache_2,
        offs_t,
        offs_d,
        offs_vd,
        offs_shfl,
        dim_mask,
        v_dim_mask,
        pgm.max_seq_prefix_len,
    )

    masked_tile_start = pgm.tile_start
    if SPLIT_UNMASKED_LOOP:
        M, L, acc = attention_loop(
            pgm, kv_loader, M, L, acc, pgm.tile_start, pgm.unmasked_tile_end, False
        )
        masked_tile_start = pgm.unmasked_tile_end
    M, L, acc = attention_loop(
        pgm, kv_loader, M, L, acc, masked_tile_start, pgm.tile_end, True
    )

    # epilogue
    if NUM_SEGMENTS_PER_SEQ > 1:
        pgm.store_partial(
            acc, M, L, segm_output_ptr, segm_max_ptr, segm_expsum_ptr, segm_idx
        )
    else:
        pgm.store_output(
            acc,
            L,
            output_ptr,
            output_stride_0,
            output_stride_1,
            out_scale_ptr,
            FP8_MIN,
            FP8_MAX,
        )


_reduce_segments_repr = make_kernel_repr(
    "reduce_segments",
    [
        "num_query_heads",
        "TILE_SIZE",
        "HEAD_SIZE",
        "V_HEAD_SIZE",
        "NUM_SEGMENTS_PER_SEQ",
    ],
)


@triton.jit(repr=_reduce_segments_repr)
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_ptr,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    V_HEAD_SIZE: tl.constexpr,  # int, value head size (may differ from HEAD_SIZE)
    V_HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    out_scale = None
    if out_scale_ptr is not None:
        out_scale = 1 / tl.load(out_scale_ptr)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

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

    offs_vd = tl.arange(0, V_HEAD_SIZE_PADDED)
    if V_HEAD_SIZE_PADDED != V_HEAD_SIZE:
        v_dim_mask = offs_vd < V_HEAD_SIZE
    else:
        v_dim_mask = tl.full((1,), 1, dtype=tl.int1)

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
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * V_HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * V_HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * V_HEAD_SIZE_PADDED
        + offs_vd[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & v_dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.math.exp2(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if out_scale_ptr is not None:
        acc = acc * out_scale

    if output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0 + query_head_idx * output_stride_1 + offs_vd
    )
    tl.store(
        output_ptr + output_offset,
        acc.to(output_ptr.type.element_ty),
        mask=v_dim_mask,
    )
