# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import functools
import math

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.mla import (
    _mla_decode_fwd_kernel as triton_mla_decode_fwd_kernel,
)
from aiter.ops.triton._triton_kernels.attention.mla import (
    _mla_decode_fwd_reduce_kernel as triton_mla_decode_fwd_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.attention.mla import (
    _mla_prefill_fwd_kernel as triton_mla_prefill_fwd_kernel,
)
from aiter.ops.triton.utils.device_info import get_num_sms

try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.mla import (
        _mla_decode_fwd_kernel as gluon_mla_decode_fwd_kernel,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.mla import (
        _mla_decode_fwd_kernel_non_pipelined as gluon_mla_decode_fwd_kernel_non_pipelined,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.mla import (
        _mla_decode_fwd_reduce_kernel as gluon_mla_decode_fwd_reduce_kernel,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.mla import (
        _mla_prefill_fwd_kernel_non_pipelined as gluon_mla_prefill_fwd_kernel_non_pipelined,
    )
except:  # noqa: E722
    gluon_mla_prefill_fwd_kernel_non_pipelined = None
    gluon_mla_decode_fwd_kernel_non_pipelined = None
    gluon_mla_decode_fwd_kernel = None
    gluon_mla_decode_fwd_reduce_kernel = None

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.types import e4m3_dtype

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64


def select_2d_config(
    block_size,
    head_size,
    max_seqlen_k,
    num_queries_per_kv,
    num_2d_prgms,
):
    TILE_SIZE = block_size
    num_stages_2d = 1
    num_warps = 8

    return {
        "TILE_SIZE": TILE_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages_2d,
        "waves_per_eu": 1,
    }


def select_3d_config(
    block_size,
    max_seqlen_k,
    target_num_prgms,
    num_2d_prgms,
    q_dtype,
    kv_dtype,
    shuffled_kv_cache,
    BLOCK_M,
):
    attn_num_warps = 2
    reduce_num_warps = 2
    attn_waves_per_eu = 1
    reduce_waves_per_eu = 2
    num_segments = 0
    TILE_SIZE = block_size
    if IS_DEVICE_ARCH_GFX12:
        # If we cannot infer max_seqlen_k during graph capture
        maybe_guess_max_seqlen_k = 128000 if max_seqlen_k == 0 else max_seqlen_k
        if BLOCK_M > 32:
            attn_num_warps = 4
        reduce_num_warps = 4
        attn_waves_per_eu = 1
        reduce_waves_per_eu = 1
        if shuffled_kv_cache and kv_dtype == torch.uint8:
            assert (
                block_size == 128
            ), "Only block_size == 128 is supported for FP4 KV cache"

        occ = attn_waves_per_eu * 4 // attn_num_warps
        MAX_SEGMENTS = max(1, math.ceil(maybe_guess_max_seqlen_k / TILE_SIZE))
        num_segments = max(1, target_num_prgms // 4 * occ // max(1, num_2d_prgms))
        num_segments = min(MAX_SEGMENTS, num_segments)
        num_segments = triton.next_power_of_2(num_segments)

    MAX_SEGMENTS = min(128, math.ceil(max_seqlen_k / TILE_SIZE))
    if num_segments == 0:
        num_segments = math.ceil(target_num_prgms / num_2d_prgms) * 2
        num_segments = min(num_segments, MAX_SEGMENTS)
        num_segments = triton.next_power_of_2(num_segments)
        num_segments = min(num_segments, 128)
        MIN_SEGMENTS = max(8, num_segments)
        num_segments = max(num_segments, MIN_SEGMENTS)

        if num_segments == MIN_SEGMENTS:
            reduce_num_warps = 1

    attn_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": attn_num_warps,
        "waves_per_eu": attn_waves_per_eu,
        "num_stages": 2 if DEVICE_ARCH in ("gfx1250", "gfx950") else 1,
    }
    reduce_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": reduce_num_warps,
        "waves_per_eu": reduce_waves_per_eu,
        "num_stages": 1,
    }
    return attn_config, reduce_config


def mla_prefill_fwd(
    q,  # [num_tokens_per_seq * num_seqs, num_query_heads, qk_lora_rank + qk_rope_head_dim]
    kv_buffer,  # [num_blocks, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
    out,
    cu_seqlens_q,  # [num_seqs + 1]
    seqused_k,  # [num_seqs]
    max_seqlen_kv: int,
    block_tables,  # [batch_size, max_num_blocks_per_seq]
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    causal: bool,
    q_descale,
    kv_descale,
    out_scale=None,
    shuffled_kv_cache: bool = False,
):
    assert causal, "Only causal attention is supported"
    assert (
        not shuffled_kv_cache
    ), "Shuffled kv cache is not supported in mla_prefill_fwd"

    _total_num_tokens, num_query_heads, qk_head_dim = q.shape
    _num_blocks, block_size, num_kv_heads, _ = kv_buffer.shape
    num_seqs = len(seqused_k)
    num_queries_per_kv = num_query_heads // num_kv_heads
    q_dtype = q.dtype
    kv_buffer_dtype = kv_buffer.dtype
    K_WIDTH = 16 if kv_buffer_dtype == e4m3_dtype else 8
    QUERY_DTYPE = "fp8" if q_dtype == e4m3_dtype else "bf16"
    KV_CACHE_DTYPE = "fp8" if kv_buffer_dtype == e4m3_dtype else "bf16"

    assert (
        kv_lora_rank + qk_rope_head_dim == qk_head_dim
    ), "qk_head_dim must be equal to kv_lora_rank + qk_rope_head_dim"

    # BLOCK_M = 128
    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    assert BLOCK_Q >= 1 or (num_queries_per_kv > BLOCK_M)
    BLOCK_Q = max(BLOCK_Q, 1)
    # When num_queries_per_kv > BLOCK_M the query heads of a single KV head do
    # not fit into one BLOCK_M tile, so we split them across NUM_HEAD_BLOCKS
    # blocks along the head dimension.
    NUM_HEAD_BLOCKS = (num_queries_per_kv + BLOCK_M - 1) // BLOCK_M
    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    # cu_count = get_num_sms()
    total_num_q_blocks = (q.shape[0] // BLOCK_Q + num_seqs) * NUM_HEAD_BLOCKS
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    # if batch contains a prefill
    attn_config = select_2d_config(
        block_size,
        kv_lora_rank,
        max_seqlen_kv,
        num_queries_per_kv,
        num_2d_prgms,
    )

    if IS_DEVICE_ARCH_GFX12:
        gluon_mla_prefill_fwd_kernel_non_pipelined[(num_kv_heads, total_num_q_blocks)](
            output_ptr=out,
            query_ptr=q,
            kv_buffer_ptr=kv_buffer,
            block_tables_ptr=block_tables,
            seq_lens_ptr=seqused_k,
            SCALE=softmax_scale,
            q_scale_ptr=q_descale,
            kv_scale_ptr=kv_descale,
            out_scale_ptr=out_scale,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            block_tables_stride=block_tables.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            KV_LORA_RANK=kv_lora_rank,
            QK_ROPE_HEAD_DIM=qk_rope_head_dim,
            stride_kv_buffer_0=kv_buffer.stride(0),
            stride_kv_buffer_1=kv_buffer.stride(1),
            stride_kv_buffer_2=kv_buffer.stride(2),
            stride_kv_buffer_3=kv_buffer.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            num_seqs=num_seqs,
            BLOCK_Q=BLOCK_Q,
            BLOCK_M=BLOCK_M,
            NUM_HEAD_BLOCKS=NUM_HEAD_BLOCKS,
            WARP_SIZE=WARP_SIZE,
            QUERY_DTYPE=QUERY_DTYPE,
            KV_CACHE_DTYPE=KV_CACHE_DTYPE,
            K_WIDTH=K_WIDTH,
            **attn_config,
        )
    else:
        triton_mla_prefill_fwd_kernel[(num_kv_heads, total_num_q_blocks)](
            output_ptr=out,
            query_ptr=q,
            kv_buffer_ptr=kv_buffer,
            block_tables_ptr=block_tables,
            seq_lens_ptr=seqused_k,
            scale=softmax_scale,
            q_scale_ptr=q_descale,
            kv_scale_ptr=kv_descale,
            out_scale_ptr=out_scale,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            block_tables_stride=block_tables.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            KV_LORA_RANK=kv_lora_rank,
            QK_ROPE_HEAD_DIM=qk_rope_head_dim,
            stride_kv_buffer_0=kv_buffer.stride(0),
            stride_kv_buffer_1=kv_buffer.stride(1),
            stride_kv_buffer_2=kv_buffer.stride(2),
            stride_kv_buffer_3=kv_buffer.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            num_seqs=num_seqs,
            BLOCK_Q=BLOCK_Q,
            BLOCK_M=BLOCK_M,
            NUM_HEAD_BLOCKS=NUM_HEAD_BLOCKS,
            **attn_config,
        )
    return out


def _fits_int32_offsets(tensor):
    # Kernels receive a pointer to the view origin, so storage_offset is not
    # part of the relative index. numel alone does not bound a strided view.
    strides = tensor.stride()
    return (
        all(0 <= stride < 2**31 for stride in strides)
        and sum((size - 1) * stride for size, stride in zip(tensor.shape, strides))
        < 2**31
    )


# Tuned decode rules: configs/<arch>/gluon/attention/mla_decode/DEFAULT.json.
# A rule depends only on the batch size, so a captured CUDA graph keeps its
# launch config. Other arches and shapes use select_3d_config.


def _decode_dtype_tag(dtype):
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == e4m3_dtype:
        return "fp8"
    if dtype == torch.uint8:
        return "nvfp4"
    return str(dtype)


@functools.lru_cache(maxsize=1)
def _tuned_decode_rules():
    if not IS_DEVICE_ARCH_GFX12:
        return ()
    cfg_dir = resolve_config_dir("attention", "MLA_DECODE", backend="gluon")
    table = load_config_json(f"{cfg_dir}/DEFAULT.json", required=False)
    return tuple(table["decode"]) if table else ()


def _find_tuned_decode_rule(
    num_query_heads,
    num_kv_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    q_dtype,
    kv_dtype,
):
    q_tag, kv_tag = _decode_dtype_tag(q_dtype), _decode_dtype_tag(kv_dtype)
    queries_per_kv = num_query_heads // max(1, num_kv_heads)
    for rule in _tuned_decode_rules():
        m = rule["match"]
        if (
            m["QUERY_DTYPE"] == q_tag
            and m["KV_CACHE_DTYPE"] == kv_tag
            and m["KV_LORA_RANK"] == kv_lora_rank
            and m["QK_ROPE_HEAD_DIM"] == qk_rope_head_dim
            and m["NUM_KV_HEADS"] == num_kv_heads
            and m["MIN_QUERIES_PER_KV"] <= queries_per_kv <= m["MAX_QUERIES_PER_KV"]
        ):
            return rule
    return None


def has_tuned_mla_decode_config(
    num_query_heads: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> bool:
    """Whether this arch ships a tuned single-token decode rule for the shape
    (non-shuffled KV cache; see configs/<arch>/gluon/attention/mla_decode)."""
    return (
        _find_tuned_decode_rule(
            num_query_heads,
            num_kv_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            q_dtype,
            kv_dtype,
        )
        is not None
    )


def _tuned_num_segments(segments, num_seqs, num_head_blocks):
    # Smallest power of two whose grid fills one wave of workgroups with the
    # last wave at least MIN_LAST_WAVE_FILL full; else the best-filled count.
    wave = get_num_sms() * segments["WORKGROUPS_PER_CU"]
    best, best_fill = 1, -1.0
    n = 1
    while n <= segments["MAX_SEGMENTS"]:
        prgms = num_seqs * num_head_blocks * n
        fill = prgms / (wave * triton.cdiv(prgms, wave))  # last-wave occupancy
        if prgms >= wave and fill >= segments["MIN_LAST_WAVE_FILL"]:
            return n
        if fill > best_fill + 1e-9:
            best, best_fill = n, fill
        n *= 2
    return best if num_seqs > 1 else segments["SINGLE_SEQ_SEGMENTS"]


def _tuned_decode_config(rule, num_seqs, queries_per_kv):
    """(BLOCK_M, attn_config, reduce_config) from a tuned rule for a batch of
    num_seqs single-token decode rows."""
    attn = select_leq_config(rule["attn"], axes=("B",), B=num_seqs)
    block_m = attn.pop("BLOCK_M")
    num_segments = _tuned_num_segments(
        rule["segments"], num_seqs, triton.cdiv(queries_per_kv, block_m)
    )
    reduce = select_leq_config(rule["reduce"], axes=("NSEG",), NSEG=num_segments)
    tile = rule["TILE_SIZE"]
    attn_config = {"TILE_SIZE": tile, "NUM_SEGMENTS_PER_SEQ": num_segments, **attn}
    reduce_config = {"TILE_SIZE": tile, "NUM_SEGMENTS_PER_SEQ": num_segments, **reduce}
    return block_m, attn_config, reduce_config


def mla_decode_workspace_numel(
    total_num_tokens: int,
    num_query_heads: int,
    kv_lora_rank: int,
    num_segments: int,
) -> int:
    """fp32 elements of the ``workspace`` mla_decode_fwd needs for a split
    count (segment outputs, maxima and exp-sums); 0 when it does not split."""
    if num_segments <= 1:
        return 0
    per_segment = triton.next_power_of_2(kv_lora_rank) + 2
    return total_num_tokens * num_query_heads * num_segments * per_segment


@functools.lru_cache(maxsize=None)
def mla_decode_max_workspace_numel(
    num_query_heads: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    max_num_seqs: int,
) -> int:
    """Largest workspace the tuned rule needs for any single-token decode
    batch of 1..max_num_seqs rows (0 if no tuned rule matches). Sizing a
    persistent workspace with it keeps every batch size, and so every captured
    CUDA graph, on the same buffer."""
    rule = _find_tuned_decode_rule(
        num_query_heads,
        num_kv_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        q_dtype,
        kv_dtype,
    )
    if rule is None:
        return 0
    queries_per_kv = num_query_heads // num_kv_heads
    best = 0
    for num_seqs in range(1, max_num_seqs + 1):
        nseg = _tuned_decode_config(rule, num_seqs, queries_per_kv)[1][
            "NUM_SEGMENTS_PER_SEQ"
        ]
        best = max(
            best,
            mla_decode_workspace_numel(num_seqs, num_query_heads, kv_lora_rank, nseg),
        )
    return best


def mla_decode_fwd(
    q,  # [num_tokens_per_seq * num_seqs, num_query_heads, qk_lora_rank + qk_rope_head_dim]
    kv_buffer,  # [num_blocks, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
    out,
    cu_seqlens_q,  # [num_seqs + 1]
    seqused_k,  # [num_seqs]
    max_seqlen_kv: int,
    block_tables,  # [batch_size, max_num_blocks_per_seq]
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    causal: bool,
    q_descale,
    kv_descale,
    q_scales=None,
    out_scale=None,
    shuffled_kv_cache: bool = False,
    skip_reduce: bool = False,
    *,
    kv_indptr=None,
    num_segments=None,
    attn_config=None,
    reduce_config=None,
    workspace=None,
):
    """Split-KV MLA decode (attention kernel + segment reduce).

    Optional keyword arguments:
        kv_indptr: [num_seqs + 1] int32. Selects the page-size-1 ("flat") KV
            layout: kv_buffer is [num_tokens, 1, num_kv_heads, D],
            block_tables is the flat per-token row list (kv_indices) and
            kv_indptr its per-sequence offsets; seqused_k is unused. gfx1250,
            non-shuffled cache only. The kernel tile is then independent of
            the cache page size.
        num_segments: KV split count override (1 skips the reduce).
        attn_config / reduce_config: launch-parameter overrides merged over
            the selected configs (keys of select_3d_config's dicts; attn_config
            may also set BLOCK_M, and TILE_SIZE in the flat layout).
        cu_seqlens_q may be None for single-token decode on gfx1250: row i
            then belongs to sequence i.
        workspace: caller-owned 1-D contiguous fp32 tensor with at least
            mla_decode_workspace_numel(...) elements. Segment outputs, maxima
            and exp-sums are carved out of it instead of being allocated per
            call, so the call allocates nothing (CUDA-graph safe).

    On gfx1250, with no override, single-token decode shapes listed in
    configs/gfx1250/gluon/attention/mla_decode/DEFAULT.json use that tuned
    rule instead of select_3d_config. Rows with a zero KV length produce a
    zero output.
    """
    assert causal, "Only causal attention is supported"
    q_dtype = q.dtype
    kv_buffer_dtype = kv_buffer.dtype
    total_num_tokens, num_query_heads, qk_head_dim = q.shape

    BLOCK_SCALES_SIZE = 16
    if q_dtype == torch.uint8:
        # A4W4
        assert q_scales is not None and q_scales.dtype == e4m3_dtype
        qk_head_dim = qk_head_dim * 2
        QUERY_DTYPE = "nvfp4"
    elif q_dtype == e4m3_dtype:
        QUERY_DTYPE = "fp8"
    else:
        QUERY_DTYPE = "bf16"

    if kv_buffer_dtype == torch.uint8:
        # A8W4 A4W4
        assert IS_DEVICE_ARCH_GFX12, "FP4 KV cache is only supported on GFX12"
        KV_CACHE_DTYPE = "nvfp4"
    elif kv_buffer_dtype == e4m3_dtype:
        KV_CACHE_DTYPE = "fp8"
    else:
        KV_CACHE_DTYPE = "bf16"

    SCALE_K_WIDTH_LORA = 0
    SCALE_K_WIDTH_ROPE = 0
    if shuffled_kv_cache:
        SCALE_K_WIDTH_LORA = 4
        SCALE_K_WIDTH_ROPE = 4
        if kv_buffer_dtype == torch.uint8:
            num_blocks, num_kv_heads, block_size, _ = kv_buffer.shape
            K_WIDTH = 16
            SCALE_K_LORA = kv_lora_rank // 16
            SCALE_K_ROPE = qk_rope_head_dim // 16
            SCALE_K_WIDTH_LORA = (
                min(16, triton.next_power_of_2(SCALE_K_LORA))
                if SCALE_K_LORA >= 4
                else SCALE_K_LORA
            )
            SCALE_K_WIDTH_ROPE = (
                min(16, triton.next_power_of_2(SCALE_K_ROPE))
                if SCALE_K_ROPE >= 4
                else SCALE_K_ROPE
            )
        else:
            num_blocks, num_kv_heads, block_size, _ = kv_buffer.shape
            K_WIDTH = 16 if kv_buffer_dtype == e4m3_dtype else 8
    else:
        num_blocks, block_size, num_kv_heads, _ = kv_buffer.shape
        K_WIDTH = 16 if kv_buffer_dtype == e4m3_dtype else 8

    kv_indptr_table = kv_indptr is not None
    if kv_indptr_table:
        assert (
            IS_DEVICE_ARCH_GFX12 and not shuffled_kv_cache
        ), "kv_indptr (page-size-1 KV) needs the gfx1250 non-shuffled kernel"
        assert block_size == 1, "kv_indptr expects kv_buffer [num_tokens, 1, H_kv, D]"
        assert block_tables.dim() == 1 and block_tables.stride(0) == 1
        assert num_blocks >= 128, "page-size-1 KV needs at least one tile of rows"
        num_seqs = kv_indptr.shape[0] - 1
        seq_lens_arg = kv_indptr
        block_tables_stride = 0
    else:
        num_seqs = len(seqused_k)
        seq_lens_arg = seqused_k
        block_tables_stride = block_tables.stride(0)
    num_tokens_per_seq = total_num_tokens // num_seqs
    num_queries_per_kv = num_query_heads // num_kv_heads
    assert cu_seqlens_q is not None or (
        IS_DEVICE_ARCH_GFX12 and not shuffled_kv_cache and num_tokens_per_seq == 1
    ), "cu_seqlens_q=None needs single-token decode on the gfx1250 kernel"

    assert (
        kv_lora_rank + qk_rope_head_dim == qk_head_dim
    ), "qk_head_dim must be equal to kv_lora_rank + qk_rope_head_dim"

    MAX_BLOCK_M = 128 if kv_buffer_dtype == e4m3_dtype else 64
    if num_queries_per_kv <= 16:
        BLOCK_M = 16
    else:
        BLOCK_M = min(triton.next_power_of_2(num_queries_per_kv), MAX_BLOCK_M)

    tuned = None
    if (
        num_segments is None
        and attn_config is None
        and reduce_config is None
        and not shuffled_kv_cache
        and num_tokens_per_seq == 1
    ):
        rule = _find_tuned_decode_rule(
            num_query_heads,
            num_kv_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            q_dtype,
            kv_buffer_dtype,
        )
        # A paged cache fixes the tile to its block size.
        if rule is not None and (kv_indptr_table or block_size == rule["TILE_SIZE"]):
            tuned = _tuned_decode_config(rule, num_seqs, num_queries_per_kv)
            BLOCK_M = tuned[0]
    if attn_config is not None and "BLOCK_M" in attn_config:
        BLOCK_M = attn_config["BLOCK_M"]
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    assert BLOCK_Q >= 1 or (num_queries_per_kv > BLOCK_M)
    BLOCK_Q = max(BLOCK_Q, 1)
    NUM_HEAD_BLOCKS = (num_queries_per_kv + BLOCK_M - 1) // BLOCK_M
    cu_count = get_num_sms()
    target_num_prgms = cu_count * 4
    ALL_DECODE = num_tokens_per_seq == 1
    if ALL_DECODE:
        total_num_q_blocks = num_seqs * NUM_HEAD_BLOCKS
    else:
        total_num_q_blocks = (
            ((num_tokens_per_seq + BLOCK_Q - 1) // BLOCK_Q) * num_seqs * NUM_HEAD_BLOCKS
        )
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    # if batch contains a prefill

    attn_overrides, reduce_overrides = attn_config, reduce_config
    if tuned is not None:
        _, attn_config, reduce_config = tuned
    else:
        attn_config, reduce_config = select_3d_config(
            # The flat layout has no page size; tile like a 64-token cache.
            64 if kv_indptr_table else block_size,
            max_seqlen_kv,
            target_num_prgms,
            num_2d_prgms,
            q_dtype,
            kv_buffer_dtype,
            shuffled_kv_cache,
            BLOCK_M,
        )
    if attn_overrides is not None:
        attn_config.update({k: v for k, v in attn_overrides.items() if k != "BLOCK_M"})
        reduce_config["TILE_SIZE"] = attn_config["TILE_SIZE"]
    if reduce_overrides is not None:
        reduce_config.update(reduce_overrides)
    if num_segments is not None:
        attn_config["NUM_SEGMENTS_PER_SEQ"] = num_segments
        reduce_config["NUM_SEGMENTS_PER_SEQ"] = num_segments
    assert kv_indptr_table or attn_config["TILE_SIZE"] == block_size or (
        shuffled_kv_cache
    ), "a paged KV cache needs TILE_SIZE == block_size"

    selected_num_segments = attn_config["NUM_SEGMENTS_PER_SEQ"]
    use_lds_pipeline = (
        DEVICE_ARCH == "gfx1250"
        and ALL_DECODE
        and total_num_tokens == num_seqs
        and shuffled_kv_cache
        and QUERY_DTYPE == "fp8"
        and KV_CACHE_DTYPE == "fp8"
        and kv_lora_rank == 512
        and qk_rope_head_dim == 64
        and block_size == 64
        and num_query_heads == 128 * num_kv_heads
        and NUM_HEAD_BLOCKS == 1
        and (
            selected_num_segments == 1
            # Zero means unknown, not a short context. Preserve partial-output
            # semantics when skip_reduce requests multiple segments.
            or (not skip_reduce and 0 < max_seqlen_kv <= 3 * block_size)
        )
        and q_scales is None
        and not kv_indptr_table
        and attn_overrides is None
        and num_segments is None
        and q.stride(2) == 1
        and kv_buffer.stride(3) == 1
        and 0 < kv_buffer.stride(1) < 2**31
        and num_blocks * num_kv_heads < 2**31
        and out.stride(2) == 1
        and block_tables.stride(1) == 1
        and _fits_int32_offsets(q)
        and _fits_int32_offsets(out)
        and _fits_int32_offsets(block_tables)
    )
    NUM_SEGMENTS = 1 if use_lds_pipeline else selected_num_segments
    if NUM_SEGMENTS > 1 and workspace is not None:
        numel = mla_decode_workspace_numel(
            total_num_tokens, num_query_heads, kv_lora_rank, NUM_SEGMENTS
        )
        assert (
            workspace.dtype == torch.float32
            and workspace.dim() == 1
            and workspace.is_contiguous()
            and workspace.numel() >= numel
        ), f"workspace must be a contiguous fp32 vector of >= {numel} elements"
        n_out = (
            total_num_tokens
            * num_query_heads
            * NUM_SEGMENTS
            * triton.next_power_of_2(kv_lora_rank)
        )
        n_stat = total_num_tokens * num_query_heads * NUM_SEGMENTS
        segm_output = workspace[:n_out].view(
            total_num_tokens,
            num_query_heads,
            NUM_SEGMENTS,
            triton.next_power_of_2(kv_lora_rank),
        )
        segm_max = workspace[n_out : n_out + n_stat].view(
            total_num_tokens, num_query_heads, NUM_SEGMENTS
        )
        segm_expsum = workspace[n_out + n_stat : n_out + 2 * n_stat].view(
            total_num_tokens, num_query_heads, NUM_SEGMENTS
        )
    elif NUM_SEGMENTS > 1:
        segm_output = torch.empty(
            total_num_tokens,
            num_query_heads,
            NUM_SEGMENTS,
            triton.next_power_of_2(kv_lora_rank),
            dtype=torch.float32,
            device=q.device,
        )
        segm_max = torch.empty(
            total_num_tokens,
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
        segm_expsum = torch.empty(
            total_num_tokens,
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        segm_output = out
        segm_max = out  # dummy ptr
        segm_expsum = out  # dummy ptr

    if use_lds_pipeline:
        attn_config.update(
            NUM_SEGMENTS_PER_SEQ=1,
            num_warps=4,
            num_stages=4,
            waves_per_eu=1,
            USE_LDS_PIPELINE=True,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            # TDM needs aligned BF16 rows. Other output views use direct stores.
            TDM_STORE=(
                out.dtype == torch.bfloat16
                and out.data_ptr() % 16 == 0
                and out.stride(0) % 8 == 0
                and out.stride(1) % 8 == 0
            ),
        )
    if IS_DEVICE_ARCH_GFX12:
        if shuffled_kv_cache:
            impl = gluon_mla_decode_fwd_kernel
        else:
            impl = gluon_mla_decode_fwd_kernel_non_pipelined
            if kv_indptr_table:
                attn_config["KV_INDPTR_TABLE"] = True

        impl[(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            query_ptr=q,
            query_scales_ptr=q_scales,
            kv_buffer_ptr=kv_buffer,
            block_tables_ptr=block_tables,
            seq_lens_ptr=seq_lens_arg,
            SCALE=softmax_scale,
            q_scale_ptr=q_descale,
            kv_scale_ptr=kv_descale,
            out_scale_ptr=(
                out_scale if (out_scale is not None and NUM_SEGMENTS == 1) else None
            ),
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            block_tables_stride=block_tables_stride,
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            query_scales_stride_0=q_scales.stride(0) if q_scales is not None else 0,
            query_scales_stride_1=q_scales.stride(1) if q_scales is not None else 0,
            KV_LORA_RANK=kv_lora_rank,
            QK_ROPE_HEAD_DIM=qk_rope_head_dim,
            stride_kv_buffer_0=kv_buffer.stride(0),
            stride_kv_buffer_1=kv_buffer.stride(1),
            stride_kv_buffer_2=kv_buffer.stride(2),
            stride_kv_buffer_3=kv_buffer.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            num_tokens_per_seq=num_tokens_per_seq,
            num_blocks=num_blocks,
            WARP_SIZE=WARP_SIZE,
            BLOCK_Q=BLOCK_Q,
            BLOCK_M=BLOCK_M,
            ALL_DECODE=ALL_DECODE,
            SHUFFLED_KV_CACHE=shuffled_kv_cache,
            K_WIDTH=K_WIDTH,
            SCALE_K_WIDTH_LORA=SCALE_K_WIDTH_LORA,
            SCALE_K_WIDTH_ROPE=SCALE_K_WIDTH_ROPE,
            QUERY_DTYPE=QUERY_DTYPE,
            KV_CACHE_DTYPE=KV_CACHE_DTYPE,
            BLOCK_SCALES_SIZE=BLOCK_SCALES_SIZE,
            NUM_HEAD_BLOCKS=NUM_HEAD_BLOCKS,
            **attn_config,
        )
    else:
        triton_mla_decode_fwd_kernel[(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            query_ptr=q,
            query_scales_ptr=q_scales,
            kv_buffer_ptr=kv_buffer,
            block_tables_ptr=block_tables,
            seq_lens_ptr=seq_lens_arg,
            scale=softmax_scale,
            q_scale_ptr=q_descale,
            kv_scale_ptr=kv_descale,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            block_tables_stride=block_tables_stride,
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            query_scales_stride_0=q_scales.stride(0) if q_scales is not None else 0,
            query_scales_stride_1=q_scales.stride(1) if q_scales is not None else 0,
            KV_LORA_RANK=kv_lora_rank,
            QK_ROPE_HEAD_DIM=qk_rope_head_dim,
            stride_kv_buffer_0=kv_buffer.stride(0),
            stride_kv_buffer_1=kv_buffer.stride(1),
            stride_kv_buffer_2=kv_buffer.stride(2),
            stride_kv_buffer_3=kv_buffer.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            num_tokens_per_seq=num_tokens_per_seq,
            BLOCK_Q=BLOCK_Q,
            BLOCK_M=BLOCK_M,
            NUM_HEAD_BLOCKS=NUM_HEAD_BLOCKS,
            ALL_DECODE=ALL_DECODE,
            SHUFFLED_KV_CACHE=shuffled_kv_cache,
            IS_Q_FP8=(q_dtype == e4m3_dtype),
            IS_KV_FP8=(kv_buffer_dtype == e4m3_dtype),
            **attn_config,
        )

    if NUM_SEGMENTS == 1:
        return segm_output
    elif skip_reduce:
        return segm_output, segm_max, segm_expsum

    # Temporarily disable gluon reduce kernel, optimize later
    # if IS_DEVICE_ARCH_GFX12:
    #     _reduce_kernel = gluon_mla_decode_fwd_reduce_kernel
    # else:
    #     _reduce_kernel = triton_mla_decode_fwd_reduce_kernel

    _reduce_kernel = triton_mla_decode_fwd_reduce_kernel

    _reduce_kernel[(total_num_tokens, num_query_heads)](
        output_ptr=out,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=seq_lens_arg,
        out_scale_ptr=out_scale,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_tables_stride=block_tables_stride,
        num_tokens_per_seq=num_tokens_per_seq,
        total_num_tokens=total_num_tokens,
        KV_LORA_RANK=kv_lora_rank,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        ALL_DECODE=ALL_DECODE,
        KV_INDPTR_TABLE=kv_indptr_table,
        **reduce_config,
    )
    return out
