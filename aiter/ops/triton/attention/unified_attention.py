# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import triton
import torch
from aiter.ops.triton.utils.device_info import get_num_sms
import math
from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    kernel_unified_attention_2d,
    kernel_unified_attention_3d,
    reduce_segments,
)
from aiter.ops.triton.gluon.unified_attention_3d_kernel import (
    _unified_attention_gluon_kernel_3d,
)
import aiter.ops.triton.utils._triton.arch_info as arch_info

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64
WAPR_SIZE_LOG2 = int(math.log2(WARP_SIZE))
from aiter.ops.triton.utils.types import e4m3_dtype

from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import get_arch


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
    arch = get_arch()

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )

    TILE_SIZE = 32 if arch.name == "gfx1201" else 16 if arch.is_rdna else 64
    waves_per_eu = 8 if arch.name == "gfx1151" else 6 if arch.is_rdna else 2

    max_num_stages_2d = 2 if head_size > 128 else 4

    # base prefill, for short cases
    if not all_decode:
        num_stages_2d, num_warps = 1, 2
    # pure decode config
    else:
        # to not have masking when loading KV
        TILE_SIZE = min(64, triton.next_power_of_2(block_size))
        if arch.is_rdna:
            num_stages_2d, num_warps = 1, 4
        else:
            num_stages_2d, num_warps = 3, 2

    # large prefill config
    if max_seqlen_q >= 256:
        BLOCK_M = 64 if arch.is_rdna else 128
        num_stages_2d, num_warps = 1, 4

    BLOCK_Q = BLOCK_M // num_queries_per_kv
    num_stages_2d = min(max_num_stages_2d, num_stages_2d)

    return {
        "BLOCK_M": BLOCK_M,
        "BLOCK_Q": BLOCK_Q,
        "TILE_SIZE": TILE_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages_2d,
        "waves_per_eu": waves_per_eu,
    }


def select_3d_config(
    head_size,
    block_size,
    max_seqlen_k,
    target_num_prgms,
    num_2d_prgms,
    q_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    shuffled_kv_cache: bool = False,
):
    reduce_num_warps = 2
    attn_warps = 2
    if shuffled_kv_cache:
        if kv_cache_dtype == torch.bfloat16 and block_size == 64:
            attn_warps = 1
        if (
            q_dtype == torch.bfloat16
            and kv_cache_dtype == e4m3_dtype
            and block_size == 128
        ):
            attn_warps = 1

        assert (
            block_size >= 64
        ), "Only block_size >= 64 is supported for shuffled KV cache"

    TILE_SIZE = min(64, triton.next_power_of_2(block_size))

    MAX_SEGMENTS = min(128, math.ceil(max_seqlen_k / TILE_SIZE))
    num_segments = math.ceil(target_num_prgms / num_2d_prgms) * 2
    num_segments = min(num_segments, MAX_SEGMENTS)
    num_segments = triton.next_power_of_2(num_segments)
    num_segments = min(num_segments, 128)
    MIN_SEGMENTS = min(8, MAX_SEGMENTS)
    num_segments = max(num_segments, MIN_SEGMENTS)

    if num_segments == MIN_SEGMENTS:
        reduce_num_warps = 1

    attn_stages = 2

    total_num_wg = num_2d_prgms * num_segments
    waves_per_eu = 2
    occ = waves_per_eu * 4 // attn_warps

    if total_num_wg < occ * target_num_prgms:
        # occ too high, increase attn_warps to relax occ
        attn_warps = (waves_per_eu * 4) // max(
            1, triton.next_power_of_2(total_num_wg // target_num_prgms)
        )
        attn_warps = max(attn_warps, 1)
        attn_warps = min(attn_warps, 4)

    if (
        2 * attn_stages * TILE_SIZE * head_size * kv_cache_dtype.itemsize * occ
        >= 327680
    ):
        waves_per_eu = 0

    attn_config = {
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": attn_warps,
        "num_stages": attn_stages,
        "waves_per_eu": waves_per_eu,
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


def unified_attention(
    q,
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
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    shuffled_kv_cache: bool = False,
    skip_reduce: bool = False,
):
    assert causal, "Only causal attention is supported"
    if not IS_DEVICE_ARCH_GFX12:
        assert q_descale is None, "Q scales not supported for Triton kernel on gfx950"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None
    SLIDING_WINDOW = 1 + window_size[0]

    q_dtype = q.dtype
    kv_cache_dtype = k.dtype
    num_tokens, num_query_heads, head_size = q.shape
    if shuffled_kv_cache:
        # key_cache: num_blocks, num_kv_heads, block_size, head_size
        # value_cache: num_blocks, num_kv_heads, head_size, block_size
        num_blocks, num_kv_heads, block_size, _ = k.shape
        # block_size = block_size * 16
    else:
        # key_cache and value_cache: num_blocks, block_size, num_kv_heads, head_size
        num_blocks, block_size, num_kv_heads, _ = k.shape
    num_seqs = len(seqused_k)
    num_queries_per_kv = num_query_heads // num_kv_heads

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
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
    if IS_DEVICE_ARCH_GFX12:
        target_num_prgms = 256
    else:
        cu_count = get_num_sms()
        target_num_prgms = cu_count * 4
    ALL_DECODE = max_seqlen_q == 1
    if ALL_DECODE:
        total_num_q_blocks = num_seqs
    else:
        total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    # if batch contains a prefill
    if use_2d_kernel(
        head_size,
        SLIDING_WINDOW,
        ALL_DECODE,
        max_seqlen_q,
        max_seqlen_k,
        target_num_prgms,
        num_2d_prgms,
    ):
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
            k_scale=k_descale,
            v_scale=v_descale,
            out_scale=1 / output_scale if output_scale is not None else 1.0,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
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
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            num_seqs=num_seqs,
            USE_FP8=output_scale is not None,
            ALL_DECODE=ALL_DECODE,
            **config,
        )

    else:
        attn_config, reduce_config = select_3d_config(
            head_size,
            block_size,
            max_seqlen_k,
            target_num_prgms,
            num_2d_prgms,
            q_dtype,
            kv_cache_dtype,
            shuffled_kv_cache,
        )
        NUM_SEGMENTS = attn_config["NUM_SEGMENTS_PER_SEQ"]
        segm_output = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            triton.next_power_of_2(head_size),
            dtype=torch.float32,
            device=q.device,
        )
        segm_max = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
        segm_expsum = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )

        if IS_DEVICE_ARCH_GFX12:
            _unified_attention_gluon_kernel_3d[
                (total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)
            ](
                segm_output_ptr=segm_output,
                segm_max_ptr=segm_max,
                segm_expsum_ptr=segm_expsum,
                query_ptr=q,
                key_cache_ptr=k,
                value_cache_ptr=v,
                sink_ptr=sinks,
                block_tables_ptr=block_table,
                seq_lens_ptr=seqused_k,
                alibi_slopes_ptr=alibi_slopes,
                qq_bias_ptr=qq_bias,
                q_scale_ptr=q_descale,
                k_scale_ptr=k_descale,
                v_scale_ptr=v_descale,
                softcap=softcap,
                num_seqs=num_seqs,
                num_blocks=num_blocks,
                block_table_stride=block_table.stride(0),
                query_stride_0=q.stride(0),
                query_stride_1=q.stride(1),
                qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
                BLOCK_SIZE=block_size,
                HEAD_SIZE=head_size,
                USE_ALIBI_SLOPES=use_alibi_slopes,
                USE_QQ_BIAS=use_qq_bias,
                USE_SOFTCAP=(softcap > 0),
                USE_SINKS=(sinks is not None),
                SLIDING_WINDOW=SLIDING_WINDOW,
                stride_k_cache_0=k.stride(0),
                stride_k_cache_1=k.stride(1),
                stride_k_cache_2=k.stride(2),
                stride_k_cache_3=k.stride(3),
                stride_v_cache_0=v.stride(0),
                stride_v_cache_1=v.stride(1),
                stride_v_cache_2=v.stride(2),
                stride_v_cache_3=v.stride(3),
                query_start_len_ptr=cu_seqlens_q,
                SCALE=softmax_scale,
                NUM_QUERY_HEADS=num_query_heads,
                NUM_KV_HEADS=num_kv_heads,
                BLOCK_Q=BLOCK_Q,
                BLOCK_M=BLOCK_M,
                ALL_DECODE=ALL_DECODE,
                SHUFFLED_KV_CACHE=shuffled_kv_cache,
                WARP_SIZE=WARP_SIZE,
                NUM_BLOCKS_GATHER_PER_TILE=1,
                IS_Q_FP8=(q_dtype == e4m3_dtype),
                IS_KV_FP8=(kv_cache_dtype == e4m3_dtype),
                **attn_config,
            )
        else:
            kernel_unified_attention_3d[
                (total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)
            ](
                segm_output_ptr=segm_output,
                segm_max_ptr=segm_max,
                segm_expsum_ptr=segm_expsum,
                query_ptr=q,
                key_cache_ptr=k,
                value_cache_ptr=v,
                sink_ptr=sinks,
                block_tables_ptr=block_table,
                seq_lens_ptr=seqused_k,
                alibi_slopes_ptr=alibi_slopes,
                qq_bias_ptr=qq_bias,
                scale=softmax_scale,
                k_scale=k_descale,
                v_scale=v_descale,
                softcap=softcap,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                block_table_stride=block_table.stride(0),
                query_stride_0=q.stride(0),
                query_stride_1=q.stride(1),
                qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
                BLOCK_SIZE=block_size,
                TILE_SIZE=block_size,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                USE_QQ_BIAS=use_qq_bias,
                USE_SOFTCAP=(softcap > 0),
                USE_SINKS=(sinks is not None),
                SLIDING_WINDOW=SLIDING_WINDOW,
                stride_k_cache_0=k.stride(0),
                stride_k_cache_1=k.stride(1),
                stride_k_cache_2=k.stride(2),
                stride_k_cache_3=k.stride(3),
                stride_v_cache_0=v.stride(0),
                stride_v_cache_1=v.stride(1),
                stride_v_cache_2=v.stride(2),
                stride_v_cache_3=v.stride(3),
                query_start_len_ptr=cu_seqlens_q,
                BLOCK_Q=BLOCK_Q,
                num_seqs=num_seqs,
                BLOCK_M=BLOCK_M,
                ALL_DECODE=ALL_DECODE,
                **attn_config,
            )

        if skip_reduce:
            return segm_output, segm_max, segm_expsum

        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            USE_FP8=output_scale is not None,
            **reduce_config,
        )
