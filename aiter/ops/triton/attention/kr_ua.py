"""Tail-split schedule for the 2-D unified attention launch (gfx950 prefill).

Measured on MI355X (256 CU, gfx950) with the gpt-oss-120b shape
(64 q-heads / 8 kv-heads, d=64, page 64, sinks, 61.4k cached KV):

  q-blocks | workgroups | kernel_unified_attention_2d
     256   |    2048    |  6.55 ms
     257   |    2056    |  8.26 ms   (+26%)
     305   |    2440    |  8.11 ms   (flat all the way from 257)

i.e. ~2048 workgroups are co-resident; anything past that runs as a nearly
empty extra generation whose few stragglers are latency-bound and cost ~1.6 ms
regardless of how little work they carry.  TTFT is therefore a staircase in the
extend length, and a 4-token change can cost 12-26%.

This wrapper splits the launch:
  * head: the first `nb_full` q-blocks (a whole multiple of the co-residency
    capacity) go through the unmodified kernel;
  * tail: the remaining q-blocks run with the KV dimension split NUM_SEGMENTS
    ways so the straggler work spreads over the whole machine, then the partial
    (acc, max, expsum) triples are merged by reduce_segments_tail.

It also trims the grid to the q-blocks that actually carry work: with
breakable prefill CUDA graphs sglang pads the token count up to a capture
bucket, and the padded q-blocks are launched only to return immediately.
"""

import os

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.kr_split import (
    kernel_unified_attention_2d_split,
    reduce_segments_tail,
)
from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    kernel_unified_attention_2d,
)
from aiter.ops.triton.attention.unified_attention import (
    select_2d_config,
    unified_attention as _unified_attention_base,
    use_2d_kernel,
)
from aiter.ops.triton.utils.device_info import get_num_sms

ENABLED = bool(int(os.environ.get("KR_TAILSPLIT", "0")))
# workgroups co-resident per CU for the BLOCK_M=128 prefill config
_OCC = int(os.environ.get("KR_TS_OCC", "8"))
_MAX_SEGMENTS = int(os.environ.get("KR_TS_MAX_SEG", "64"))
# Below ~8 segments the split tail costs about what the un-split straggler
# generation costs, and the extra reduce makes it a small net loss (measured
# crossover at tail ~40-63 q-blocks), so only split a genuinely thin tail.
_MIN_SEGMENTS = int(os.environ.get("KR_TS_MIN_SEG", "8"))
_CAP_BLOCKS = int(os.environ.get("KR_TS_CAP_BLOCKS", "0"))
_TRIM_GRID = bool(int(os.environ.get("KR_TS_TRIM_GRID", "1")))
_buf_cache = {}


def _buffers(nrows, num_segments, head_size_padded, device):
    key = (nrows, num_segments, head_size_padded, device)
    buf = _buf_cache.get(key)
    if buf is None:
        buf = (
            torch.empty(
                nrows, num_segments, head_size_padded, dtype=torch.float32,
                device=device,
            ),
            torch.empty(nrows, num_segments, dtype=torch.float32, device=device),
            torch.empty(nrows, num_segments, dtype=torch.float32, device=device),
        )
        _buf_cache[key] = buf
    return buf


def plan_tail_split(active_blocks, num_kv_heads, cu_count):
    """Return (nb_full, num_tail_blocks, num_segments) or None."""
    cap_blocks = _CAP_BLOCKS or (_OCC * cu_count) // num_kv_heads
    if cap_blocks < 2 or active_blocks <= cap_blocks:
        return None
    nb_full = (active_blocks // cap_blocks) * cap_blocks
    tail = active_blocks - nb_full
    if tail <= 0:
        return None
    seg = min(_MAX_SEGMENTS, max(1, cap_blocks // tail))
    seg = 1 << (seg.bit_length() - 1)
    if seg < _MIN_SEGMENTS:
        return None
    return nb_full, tail, seg


def unified_attention_tailsplit(
    q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
    softmax_scale, causal, window_size, block_table, softcap, q_descale,
    k_descale, v_descale, q_scales=None, alibi_slopes=None, output_scale=None,
    qq_bias=None, sinks=None, shuffled_kv_cache: bool = False,
    skip_reduce: bool = False,
):
    def _fallback():
        return _unified_attention_base(
            q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
            softmax_scale, causal, window_size, block_table, softcap, q_descale,
            k_descale, v_descale, q_scales=q_scales, alibi_slopes=alibi_slopes,
            output_scale=output_scale, qq_bias=qq_bias, sinks=sinks,
            shuffled_kv_cache=shuffled_kv_cache, skip_reduce=skip_reduce,
        )

    SLIDING_WINDOW = 1 + window_size[0]
    q_dtype = q.dtype
    kv_cache_dtype = k.dtype
    num_tokens, num_query_heads, head_size = q.shape
    num_seqs = len(seqused_k)

    if (
        not causal
        or shuffled_kv_cache
        or skip_reduce
        or num_seqs != 1
        or alibi_slopes is not None
        or qq_bias is not None
        or q_scales is not None
        or output_scale is not None
        or softcap
        or SLIDING_WINDOW > 0
        or int(max_seqlen_q) == 1
        or q_dtype == torch.uint8
        or kv_cache_dtype == torch.uint8
        or k.dim() != 4
    ):
        return _fallback()

    num_blocks, block_size, num_kv_heads, _ = k.shape
    K_WIDTH = 16 if kv_cache_dtype.itemsize == 1 else 8
    num_queries_per_kv = num_query_heads // num_kv_heads

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    cu_count = get_num_sms()
    total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs
    if not use_2d_kernel(
        head_size, SLIDING_WINDOW, False, max_seqlen_q, max_seqlen_k,
        cu_count * 4, total_num_q_blocks * num_kv_heads,
    ):
        return _fallback()

    config = select_2d_config(
        block_size, head_size, SLIDING_WINDOW, False, max_seqlen_q, max_seqlen_k,
        num_queries_per_kv, total_num_q_blocks * num_kv_heads, q_dtype,
        kv_cache_dtype, shuffled_kv_cache,
    )
    BLOCK_Q = config["BLOCK_Q"]
    BLOCK_M = config["BLOCK_M"]
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    # q-blocks that actually carry work (the rest are CUDA-graph padding)
    active_blocks = (int(max_seqlen_q) + BLOCK_Q - 1) // BLOCK_Q
    active_blocks = min(active_blocks, total_num_q_blocks)

    plan = plan_tail_split(active_blocks, num_kv_heads, cu_count)
    head_blocks = active_blocks if _TRIM_GRID else total_num_q_blocks
    if plan is not None:
        head_blocks, tail, num_segments = plan
    if head_blocks <= 0:
        return _fallback()

    head_size_padded = triton.next_power_of_2(head_size)

    kernel_unified_attention_2d[(num_kv_heads, head_blocks)](
        output_ptr=out, query_ptr=q, key_cache_ptr=k, value_cache_ptr=v,
        sink_ptr=sinks, block_tables_ptr=block_table, seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=None, qq_bias_ptr=None, scale=softmax_scale,
        q_descale_ptr=q_descale, k_descale_ptr=k_descale, v_descale_ptr=v_descale,
        out_scale_ptr=None, softcap=softcap, num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0), query_stride_0=q.stride(0),
        query_stride_1=q.stride(1), output_stride_0=out.stride(0),
        output_stride_1=out.stride(1), qq_bias_stride_0=0, BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size, HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=False, USE_QQ_BIAS=False, USE_SOFTCAP=False,
        USE_SINKS=(sinks is not None), SLIDING_WINDOW=SLIDING_WINDOW,
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q, num_seqs=num_seqs, ALL_DECODE=False,
        SHUFFLED_KV_CACHE=False, K_WIDTH=K_WIDTH, **config,
    )
    if plan is None:
        return out

    nrows = tail * num_kv_heads * BLOCK_M
    segm_output, segm_max, segm_expsum = _buffers(
        nrows, num_segments, head_size_padded, q.device
    )

    kernel_unified_attention_2d_split[(num_kv_heads, tail, num_segments)](
        segm_output_ptr=segm_output, segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum, query_ptr=q, key_cache_ptr=k,
        value_cache_ptr=v, sink_ptr=sinks, block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k, scale=softmax_scale, q_descale_ptr=q_descale,
        k_descale_ptr=k_descale, v_descale_ptr=v_descale,
        num_query_heads=num_query_heads, num_queries_per_kv=num_queries_per_kv,
        num_kv_heads=num_kv_heads, block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0), query_stride_1=q.stride(1),
        BLOCK_SIZE=block_size, TILE_SIZE=config["TILE_SIZE"], HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded, USE_SINKS=(sinks is not None),
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q, BLOCK_Q=BLOCK_Q, num_seqs=num_seqs,
        BLOCK_M=BLOCK_M, q_block_offset=head_blocks, NUM_SEGMENTS=num_segments,
        num_warps=config["num_warps"], num_stages=config["num_stages"],
        waves_per_eu=config["waves_per_eu"],
    )

    reduce_segments_tail[(nrows,)](
        output_ptr=out, segm_output_ptr=segm_output, segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum, query_start_len_ptr=cu_seqlens_q,
        out_scale_ptr=None, num_seqs=num_seqs, num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv, num_kv_heads=num_kv_heads,
        output_stride_0=out.stride(0), output_stride_1=out.stride(1),
        HEAD_SIZE=head_size, HEAD_SIZE_PADDED=head_size_padded, BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M, NUM_SEGMENTS=num_segments, q_block_offset=head_blocks,
        num_warps=2, num_stages=1,
    )
    return out
