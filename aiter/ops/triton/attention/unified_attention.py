# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import triton
import aiter
import torch
from aiter.ops.triton.utils.device_info import get_num_sms
import math
from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    kernel_unified_attention_2d,
    kernel_unified_attention_3d,
    reduce_segments,
)
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (sage_quant, sage_quant_mxfp4)
from enum import IntEnum


def get_config(num_tokens, num_seqs, num_queries_per_kv, num_kv_heads, head_size, window_size, max_seqlen_q, max_seqlen_k, block_size, q_element_size):
    SLIDING_WINDOW = 1 + window_size[0]
    ALL_DECODE = max_seqlen_q == 1
    cu_count = get_num_sms()
    
    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )

    BLOCK_Q = BLOCK_M // num_queries_per_kv
    
    total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs
    target_num_prgms = cu_count * 4
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    # if use_2d_kernel(
    #     head_size,
    #     SLIDING_WINDOW,
    #     ALL_DECODE,
    #     max_seqlen_q,
    #     max_seqlen_k,
    #     target_num_prgms,
    #     num_2d_prgms,
    # ):
    return select_2d_config(
        block_size,
        head_size,
        SLIDING_WINDOW,
        ALL_DECODE,
        max_seqlen_q,
        max_seqlen_k,
        num_queries_per_kv,
        num_2d_prgms,
    )
    # else:
    #     attn_config, reduce_config = select_3d_config(
    #         head_size,
    #         block_size,
    #         q_element_size,
    #         max_seqlen_k,
    #         target_num_prgms,
    #         num_2d_prgms,
    #     )
    #     return attn_config

def select_2d_config(
    block_size,
    head_size,
    sliding_window,
    all_decode,
    max_seqlen_q,
    max_seqlen_k,
    num_queries_per_kv,
    num_2d_prgms,
):
    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    TILE_SIZE = 64
    # in case head_size is large
    max_num_stages_2d = 4
    if head_size > 128:
        max_num_stages_2d = 2
    if all_decode == False:
        num_stages_2d = 1
        num_warps = 2
    else:
        num_stages_2d = 3
        num_warps = 2
        TILE_SIZE = min(triton.next_power_of_2(block_size), 64)

    if max_seqlen_q >= 256:
        BLOCK_M = 128
        num_stages_2d = 1
        num_warps = 4
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    num_stages_2d = min(max_num_stages_2d, num_stages_2d)
    return {
        "BLOCK_M": BLOCK_M,
        "BLOCK_Q": BLOCK_Q,
        "TILE_SIZE": TILE_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages_2d,
        "waves_per_eu": 2,
    }


def select_3d_config(
    head_size, block_size, element_size, max_seqlen_k, target_num_prgms, num_2d_prgms
):
    reduce_num_warps = 2
    attn_warps = 2
    TILE_SIZE = min(triton.next_power_of_2(block_size), 64)
    # MAX_SEGMENTS = min(128, math.ceil(max_seqlen_k / TILE_SIZE))
    num_segments = math.ceil(target_num_prgms / num_2d_prgms)
    num_segments = triton.next_power_of_2(num_segments)
    num_segments = min(num_segments, 128)
    MIN_SEGMENTS = 16 if TILE_SIZE <= 16 else 8
    num_segments = max(num_segments, MIN_SEGMENTS)
    if num_segments == MIN_SEGMENTS:
        reduce_num_warps = 1
    attn_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": attn_warps,
        "num_stages": 1,
        "waves_per_eu": 2,
    }
    reduce_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": reduce_num_warps,
        "num_stages": 1,
        "waves_per_eu": 2,
    }
    return attn_config, reduce_config


def use_2d_kernel(
    head_size,
    sliding_window,
    all_decode,
    max_seqlen_q,
    max_seqlen_k,
    target_num_prgms,
    num_2d_prgms,
):
    return (
        (sliding_window > 0)
        or (max_seqlen_k <= 512)
        or (num_2d_prgms > target_num_prgms)
    )

class SAGE_VERSION(IntEnum):
    SAGE = 1
    SAGE_MXFP4 = 2

def get_sage_scale_strides(
        q_descale,
        k_descale,
        v_descale,
        num_blks,
        BLOCK_SIZE,
        TILE_SIZE,
        kv_layout,
        sage_version: SAGE_VERSION = None
    ):
    # q_descale either t,h_q,d//32 (SAGE_MXFP4) or total_num_blocks,h_k (SAGE)
    if sage_version == SAGE_VERSION.SAGE:
        stride_k_cache_scale_1, stride_k_cache_scale_2 = k_descale.stride()
        stride_k_cache_scale_3 = 0

        query_scale_stride_0, query_scale_stride_1 = q_descale.stride()
        query_scale_stride_2 = 0
    elif sage_version == SAGE_VERSION.SAGE_MXFP4:
        stride_k_cache_scale_0, stride_k_cache_scale_1, stride_k_cache_scale_2, stride_k_cache_scale_3 = k_descale.stride()
        query_scale_stride_0, query_scale_stride_1, query_scale_stride_2 = q_descale.stride()
    else:
        raise ValueError(f"Invalid sage version: {sage_version}")

    stride_v_cache_scale_0, stride_v_cache_scale_1 = v_descale.stride()

    return (
        query_scale_stride_0,
        query_scale_stride_1,
        query_scale_stride_2,
        stride_k_cache_scale_0,
        stride_k_cache_scale_1,
        stride_k_cache_scale_2,
        stride_k_cache_scale_3,
        stride_v_cache_scale_0,
        stride_v_cache_scale_1
    )


def check_quant_args_get_strides(
        q,
        q_descale,
        k,
        k_descale,
        v,
        v_descale,
        BLOCK_M,
        BLOCK_SIZE,
        TILE_SIZE,
        kv_layout,
        num_seqs=None,
        sage_version: SAGE_VERSION = None
    ):
    """
    qkv has shapes:
        - q,  # [num_tokens, num_query_heads, head_size]
        - k,  # [num_blks, blk_size, num_kv_heads, head_size]
        - v,  # [num_blks, blk_size, num_kv_heads, head_size]
    we expect the scales to have shapes:
        - q_scale,  # [num_tokens // BLOCK_M, num_kv_heads] if sage, [num_tokens, num_kv_heads, head_size // 32] if sage_mxfp4
        - k_scale,  # [num_blks * tl.cdiv(block_size, TILE_SIZE), num_kv_heads] if sage, [num_blks, blk_size, num_kv_heads, head_size // 32] if sage_mxfp4
        - v_scale,  # [num_kv_heads, head_size] if sage or sage_mxfp4
    """
    if sage_version != None:
        num_tokens, _, head_size_qk = q.shape
        if kv_layout == "thd":
            assert num_seqs is not None, "num_seqs must be provided when kv_layout is thd"
            num_blks = num_seqs
            _, num_kv_heads, head_size_v = v.shape
        elif kv_layout == "bshd":
            num_blks, _, num_kv_heads, head_size_v = v.shape
        elif kv_layout == "bhsd":
            num_blks, num_kv_heads, _, head_size_v = v.shape
        else: # cache
            num_blks, _, num_kv_heads, head_size_v = v.shape

        if sage_version == SAGE_VERSION.SAGE:
            assert q_descale.ndim == 2, f"expect q_descale to be 2D, got {q_descale.ndim}D with SAGE_VERSION={sage_version}"
            assert q_descale.shape[0] >= math.ceil(num_tokens / BLOCK_M), f"expect q_descale dim 0 >= {math.ceil(num_tokens / BLOCK_M)}, got {q_descale.shape[0]} with SAGE_VERSION={sage_version}"
            assert q_descale.shape[1] == num_kv_heads, f"expect q_descale dim 1 == {num_kv_heads}, got {q_descale.shape[1]} with SAGE_VERSION={sage_version}"

            assert k_descale.ndim == 2, f"expect k_descale to be 2D, got {k_descale.ndim}D with SAGE_VERSION={sage_version}"
            assert k_descale.shape[0] >= num_blks * math.ceil(BLOCK_SIZE / TILE_SIZE), (
                f"expect k_descale dim 0 >= {num_blks * math.ceil(BLOCK_SIZE / TILE_SIZE)}, "
                f"got {k_descale.shape[0]} with SAGE_VERSION={sage_version}"
            )
            assert v_descale.ndim == 2, f"expect v_descale to be 2D, got {v_descale.ndim}D with SAGE_VERSION={sage_version}"
            assert v_descale.shape[0] == num_kv_heads, f"expect v_descale dim 0 == {num_kv_heads}, got {v_descale.shape[0]} with SAGE_VERSION={sage_version}"
        elif sage_version == SAGE_VERSION.SAGE_MXFP4:
            head_size_qk *= 2
            expected_q_descale_shape = (num_tokens, num_kv_heads, head_size_qk // 32)
            assert q_descale.shape == expected_q_descale_shape, f"expect q_descale to have shape {expected_q_descale_shape} with SAGE_VERSION={sage_version}"
            expected_k_descale_shape = (
                *k.shape[:-1],
                head_size_qk // 32,
            )
            assert k_descale.shape == expected_k_descale_shape, f"expect k_descale to have shape {expected_k_descale_shape} with SAGE_VERSION={sage_version}"
            expected_v_descale_shape = (num_kv_heads, head_size_v)
            assert v_descale.shape == expected_v_descale_shape, f"expect v_descale to have shape {expected_v_descale_shape} with SAGE_VERSION={sage_version}"
           

        return get_sage_scale_strides(
            q_descale,
            k_descale,
            v_descale,
            num_blks,
            BLOCK_SIZE,
            TILE_SIZE,
            sage_version=sage_version,
            kv_layout=kv_layout,
        )
    # if sage_version is None, return dummy strides
    return (0, 0, 0, 0, 0, 0, 0, 0, 0)



def unified_attention(
    q, # thd layout assumed
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    block_table=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    sage_version: SAGE_VERSION = None,
    cu_seqlens_k=None, # is only needed if kv_layout is not cache and block_table is not provided as block_table=cu_seqlens_k
    kv_layout: str = "cache", # cache i.e. num_blocks, blk_size, h, d as default. Other options "bhsd", "bshd", "thd"
):
    assert causal, "Only causal attention is supported"
    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    # Extract basic dimensions
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    head_size = q.shape[2]
    block_size = max_seqlen_k if kv_layout=="thd" else v.shape[1]

    # Determine num_kv_heads based on layout
    kv_layout_dims = {
        "cache": 2,
        "thd": 1,
        "bshd": 2,
        "bhsd": 1,
    }
    assert kv_layout in kv_layout_dims, f"unsupported kv_layout: {kv_layout}"
    num_kv_heads = k.shape[kv_layout_dims[kv_layout]]

    # Setup block table
    if kv_layout == "thd":
        if block_table is None:
            assert cu_seqlens_k, "If layout is thd, cu_seqlens_k must be provided"
            block_table = cu_seqlens_k
    elif kv_layout == "cache":
        assert block_table is not None, "block_table required for cache layout"
    elif kv_layout in ["bshd", "bhsd"] and block_table is None:
        block_table = torch.arange(0, num_seqs, device=q.device, dtype=torch.int32)

    # Extract strides for k and v
    stride_k_cache = k.stride()
    stride_v_cache = v.stride()

    if kv_layout == "bhsd":
        stride_k_cache = (stride_k_cache[0], stride_k_cache[2], stride_k_cache[1], stride_k_cache[3])
        stride_v_cache = (stride_v_cache[0], stride_v_cache[2], stride_v_cache[1], stride_v_cache[3])
    elif kv_layout == "thd":
        stride_k_cache = (stride_k_cache[0], stride_k_cache[0], stride_k_cache[1], stride_k_cache[2])
        stride_v_cache = (stride_v_cache[0], stride_v_cache[0], stride_v_cache[1], stride_v_cache[2])

    stride_k_cache_0, stride_k_cache_1, stride_k_cache_2, stride_k_cache_3 = stride_k_cache
    stride_v_cache_0, stride_v_cache_1, stride_v_cache_2, stride_v_cache_3 = stride_v_cache
    block_table_stride = block_table.stride(0)

    # Configuration flags and parameters
    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None
    SLIDING_WINDOW = 1 + window_size[0]

    # Compute block and query dimensions
    num_queries_per_kv = num_query_heads // num_kv_heads
    if sage_version == SAGE_VERSION.SAGE_MXFP4:
        head_size *= 2

    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    KV_LAYOUT = {"cache": 1, "bshd": 2, "bhsd": 2, "thd": 3}[kv_layout]

    assert BLOCK_Q >= 1
    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    cu_count = get_num_sms()
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    target_num_prgms = cu_count * 4
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    ALL_DECODE = max_seqlen_q == 1
    # 3D kernel does not support sage attention yet; force 2D path when sage is active.
    # if sage_version is not None or use_2d_kernel(
    #     head_size,
    #     SLIDING_WINDOW,
    #     ALL_DECODE,
    #     max_seqlen_q,
    #     max_seqlen_k,
    #     target_num_prgms,
    #     num_2d_prgms,
    # ):
    config = select_2d_config(
        block_size,
        head_size,
        SLIDING_WINDOW,
        ALL_DECODE,
        max_seqlen_q,
        max_seqlen_k,
        num_queries_per_kv,
        num_2d_prgms,
    )

    TILE_SIZE=config["TILE_SIZE"]
    BLOCK_M=config["BLOCK_M"]
    (
        query_scale_stride_0,
        query_scale_stride_1,
        query_scale_stride_2,
        stride_k_cache_scale_0,
        stride_k_cache_scale_1,
        stride_k_cache_scale_2,
        stride_k_cache_scale_3,
        stride_v_cache_scale_0,
        stride_v_cache_scale_1 
    ) = check_quant_args_get_strides(
        q,
        q_descale,
        k,
        k_descale,
        v,
        v_descale,
        BLOCK_M,
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        sage_version=sage_version,
        num_seqs=num_seqs,
        kv_layout=kv_layout
    )

    assert config["BLOCK_Q"] >= 1
    total_num_q_blocks = q.shape[0] // config["BLOCK_Q"] + num_seqs

    
    
    kernel_unified_attention_2d[
        (
            num_kv_heads,
            total_num_q_blocks,
        )
    ](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        scale=softmax_scale,
        q_scale=q_descale,
        k_scale=k_descale,
        v_scale=v_descale,
        out_scale=1 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table_stride,
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        query_scale_stride_0=query_scale_stride_0,
        query_scale_stride_1=query_scale_stride_1,
        query_scale_stride_2=query_scale_stride_2,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=SLIDING_WINDOW,
        stride_k_cache_0=stride_k_cache_0,
        stride_k_cache_1=stride_k_cache_1,
        stride_k_cache_2=stride_k_cache_2,
        stride_k_cache_3=stride_k_cache_3,
        stride_v_cache_0=stride_v_cache_0,
        stride_v_cache_1=stride_v_cache_1,
        stride_v_cache_2=stride_v_cache_2,
        stride_v_cache_3=stride_v_cache_3,
        stride_k_cache_scale_0=stride_k_cache_scale_0,
        stride_k_cache_scale_1=stride_k_cache_scale_1,
        stride_k_cache_scale_2=stride_k_cache_scale_2,
        stride_k_cache_scale_3=stride_k_cache_scale_3,
        stride_v_cache_scale_0=stride_v_cache_scale_0,
        stride_v_cache_scale_1=stride_v_cache_scale_1,
        query_start_len_ptr=cu_seqlens_q,
        key_start_len_ptr=cu_seqlens_k,
        num_seqs=num_seqs,
        OUTPUT_FP8=output_scale is not None,
        ALL_DECODE=ALL_DECODE,
        SAGE_VERSION=sage_version,
        KV_LAYOUT=KV_LAYOUT,
        **config,
    )

    # else:
    #     attn_config, reduce_config = select_3d_config(
    #         head_size,
    #         block_size,
    #         q.element_size(),
    #         max_seqlen_k,
    #         target_num_prgms,
    #         num_2d_prgms,
    #     )
    #     NUM_SEGMENTS = attn_config["NUM_SEGMENTS_PER_SEQ"]
    #     segm_output = torch.empty(
    #         q.shape[0],
    #         num_query_heads,
    #         NUM_SEGMENTS,
    #         triton.next_power_of_2(head_size),
    #         dtype=torch.float32,
    #         device=q.device,
    #     )
    #     segm_max = torch.empty(
    #         q.shape[0],
    #         num_query_heads,
    #         NUM_SEGMENTS,
    #         dtype=torch.float32,
    #         device=q.device,
    #     )
    #     segm_expsum = torch.empty(
    #         q.shape[0],
    #         num_query_heads,
    #         NUM_SEGMENTS,
    #         dtype=torch.float32,
    #         device=q.device,
    #     )

    #     TILE_SIZE=attn_config["TILE_SIZE"]
    #     (
    #         query_scale_stride_0,
    #         query_scale_stride_1,
    #         query_scale_stride_2,
    #         stride_k_cache_scale_0,
    #         stride_k_cache_scale_1,
    #         stride_k_cache_scale_2,
    #         stride_k_cache_scale_3,
    #         stride_v_cache_scale_0,
    #         stride_v_cache_scale_1 
    #     ) = check_quant_args_get_strides(
    #         q,
    #         q_descale,
    #         k,
    #         k_descale,
    #         v,
    #         v_descale,
    #         BLOCK_M,
    #         BLOCK_SIZE=block_size,
    #         TILE_SIZE=TILE_SIZE,
    #         sage_version=sage_version
    #     )
    #     kernel_unified_attention_3d[(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
    #         segm_output_ptr=segm_output,
    #         segm_max_ptr=segm_max,
    #         segm_expsum_ptr=segm_expsum,
    #         query_ptr=q,
    #         key_cache_ptr=k,
    #         value_cache_ptr=v,
    #         sink_ptr=sinks,
    #         block_tables_ptr=block_table,
    #         seq_lens_ptr=seqused_k,
    #         alibi_slopes_ptr=alibi_slopes,
    #         qq_bias_ptr=qq_bias,
    #         scale=softmax_scale,
    #         q_scale=q_descale,
    #         k_scale=k_descale,
    #         v_scale=v_descale,
    #         softcap=softcap,
    #         num_query_heads=num_query_heads,
    #         num_queries_per_kv=num_queries_per_kv,
    #         block_table_stride=block_table.stride(0),
    #         query_stride_0=q.stride(0),
    #         query_stride_1=q.stride(1),
    #         query_scale_stride_0=query_scale_stride_0,
    #         query_scale_stride_1=query_scale_stride_1,
    #         query_scale_stride_2=query_scale_stride_2,
    #         qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
    #         BLOCK_SIZE=block_size,
    #         HEAD_SIZE=head_size,
    #         HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
    #         USE_ALIBI_SLOPES=use_alibi_slopes,
    #         USE_QQ_BIAS=use_qq_bias,
    #         USE_SOFTCAP=(softcap > 0),
    #         USE_SINKS=(sinks is not None),
    #         SLIDING_WINDOW=SLIDING_WINDOW,
    #         stride_k_cache_0=k.stride(0),
    #         stride_k_cache_1=k.stride(1),
    #         stride_k_cache_2=k.stride(2),
    #         stride_k_cache_3=k.stride(3),
    #         stride_v_cache_0=v.stride(0),
    #         stride_v_cache_1=v.stride(1),
    #         stride_v_cache_2=v.stride(2),
    #         stride_v_cache_3=v.stride(3),
    #         stride_k_cache_scale_0=stride_k_cache_scale_0,
    #         stride_k_cache_scale_1=stride_k_cache_scale_1,
    #         stride_k_cache_scale_2=stride_k_cache_scale_2,
    #         stride_k_cache_scale_3=stride_k_cache_scale_3,
    #         stride_v_cache_scale_0=stride_v_cache_scale_0,
    #         stride_v_cache_scale_1=stride_v_cache_scale_1,
    #         query_start_len_ptr=cu_seqlens_q,
    #         BLOCK_Q=BLOCK_Q,
    #         num_seqs=num_seqs,
    #         BLOCK_M=BLOCK_M,
    #         USE_DESCALE=q_descale is not None and sage_version is None,
    #         ALL_DECODE=ALL_DECODE,
    #         SAGE_VERSION=sage_version,
    #         **attn_config,
    #     )
    #     reduce_segments[(q.shape[0], num_query_heads)](
    #         output_ptr=out,
    #         segm_output_ptr=segm_output,
    #         segm_max_ptr=segm_max,
    #         segm_expsum_ptr=segm_expsum,
    #         seq_lens_ptr=seqused_k,
    #         num_seqs=num_seqs,
    #         num_query_heads=num_query_heads,
    #         out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
    #         output_stride_0=out.stride(0),
    #         output_stride_1=out.stride(1),
    #         block_table_stride=block_table.stride(0),
    #         HEAD_SIZE=head_size,
    #         HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
    #         query_start_len_ptr=cu_seqlens_q,
    #         BLOCK_Q=BLOCK_Q,
    #         OUTPUT_FP8=output_scale is not None,
    #         **reduce_config,
    #     )
