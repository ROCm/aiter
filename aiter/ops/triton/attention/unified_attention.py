# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import math
from typing import NamedTuple

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    kernel_unified_attention_2d,
    kernel_unified_attention_3d,
    reduce_segments,
)
from aiter.ops.triton.utils.device_info import get_num_sms

try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_3d import (
        _unified_attention_gluon_kernel_3d,
    )
except:  # noqa: E722
    _unified_attention_gluon_kernel_3d = None

try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_2d import (
        _unified_attention_gluon_kernel_2d,
    )
except:  # noqa: E722
    _unified_attention_gluon_kernel_2d = None

try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_reduce import (
        reduce_segments_gluon as _reduce_segments_gluon,
    )
except:  # noqa: E722
    _reduce_segments_gluon = None

from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import get_arch
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import e4m3_dtype

# Max NUM_SEGMENTS the gluon reduce holds in-thread; larger split counts fall back to the Triton reduce_segments.
_GLUON_REDUCE_MAX_SEGMENTS = 8

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64
WARP_SIZE_LOG2 = int(math.log2(WARP_SIZE))


class _UAParams(NamedTuple):
    # tensors
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    cu_seqlens_q: torch.Tensor  # [num_seqs + 1], kernels' query_start_len_ptr
    seqused_k: torch.Tensor  # [num_seqs], kernels' seq_lens_ptr
    block_table: torch.Tensor  # [num_seqs, max_num_blocks_per_seq]

    # scalars
    softmax_scale: float
    softcap: float
    causal: bool
    sliding_window: int  # kernels' SLIDING_WINDOW, i.e. 1 + window_size[0]
    max_seqlen_q: int
    max_seqlen_k: int

    # shapes
    num_tokens: int  # q.shape[0]: queries summed over all seqs
    num_query_heads: int
    num_kv_heads: int
    num_queries_per_kv: int
    head_size: int  # logical, i.e. already doubled for fp4-packed q
    num_seqs: int
    total_num_q_blocks: int  # query-block upper bound at the default BLOCK_Q
    num_2d_prgms: int  # total_num_q_blocks * num_kv_heads

    # kv cache layout
    num_blocks: int
    block_size: int  # kv page size, kernels' BLOCK_SIZE
    k_width: int
    scale_k_width: int
    block_scales_size: int  # elements sharing one quantization scale

    # dtypes and modes
    q_dtype: torch.dtype
    kv_cache_dtype: torch.dtype
    all_decode: bool  # max_seqlen_q == 1
    shuffled_kv_cache: bool
    use_alibi_slopes: bool  # alibi_slopes is not None
    use_qq_bias: bool  # qq_bias is not None

    # device
    num_sms: int  # CU count; occupancy targets are derived from it
    target_num_prgms: int  # num_sms * 4: the target the heuristics aim at

    # optional inputs
    sinks: torch.Tensor | None = None
    alibi_slopes: torch.Tensor | None = None
    qq_bias: torch.Tensor | None = None
    q_scales: torch.Tensor | None = None  # fp4 per-block query scales
    q_descale: torch.Tensor | None = None
    k_descale: torch.Tensor | None = None
    v_descale: torch.Tensor | None = None
    output_scale: torch.Tensor | None = None
    skip_reduce: bool = False


def is_2d_gluon_available(params: _UAParams):
    use_gluon_2d = (
        IS_DEVICE_ARCH_GFX12
        and _unified_attention_gluon_kernel_2d is not None
        and not params.softcap
        and not params.use_qq_bias
        and not params.use_alibi_slopes
        and params.q_dtype != torch.uint8
        and params.kv_cache_dtype != torch.uint8
        and params.q_dtype == params.kv_cache_dtype
    )
    return use_gluon_2d


def is_3d_gluon_available(params: _UAParams):
    use_gluon_3d = IS_DEVICE_ARCH_GFX12 and params.shuffled_kv_cache
    return use_gluon_3d


def is_reduce_gluon_available(
    params: _UAParams, NUM_SEGMENTS, head_size_padded, gluon_num_warps
):
    use_gluon_reduce = (
        IS_DEVICE_ARCH_GFX12
        and _reduce_segments_gluon is not None
        and params.all_decode
        and NUM_SEGMENTS <= _GLUON_REDUCE_MAX_SEGMENTS
        and head_size_padded % 32 == 0
        and params.num_query_heads % gluon_num_warps == 0
    )
    return use_gluon_reduce


def select_2d_config(
    block_size,
    head_size,
    sliding_window,
    all_decode,
    max_seqlen_q,
    max_seqlen_k,
    num_queries_per_kv,
    num_2d_prgms,
    q_dtype,
    kv_cache_dtype,
    shuffled_kv_cache,
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
        if head_size >= 512 and not arch.is_rdna:
            num_warps, num_stages_2d = 4, 2
            TILE_SIZE = 16
        elif head_size >= 256 and not arch.is_rdna:
            num_warps, num_stages_2d = 2, 2
            TILE_SIZE = 32
        else:
            # large prefill config
            if max_seqlen_q >= 256:
                BLOCK_M = 64 if arch.is_rdna else 128
                num_stages_2d, num_warps = 1, 4
            else:
                num_stages_2d, num_warps = 1, 2

    # pure decode config
    else:
        # to not have masking when loading KV
        TILE_SIZE = min(64, triton.next_power_of_2(block_size))
        if arch.is_rdna:
            num_stages_2d, num_warps = 1, 4
        else:
            if head_size >= 512:
                num_stages_2d, num_warps = 1, 4
            else:
                num_stages_2d, num_warps = 3, 2

    BLOCK_Q = BLOCK_M // num_queries_per_kv
    num_stages_2d = min(max_num_stages_2d, num_stages_2d)

    # fix TILE_SIZE to block_size if shuffled_kv_cache is True
    if shuffled_kv_cache:
        if q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:
            assert block_size >= 32, (
                "For A8W8 Unified Attention with pre-shuffled KV cache, only block_size >= 32 is supported"
            )
        TILE_SIZE = block_size
    elif q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:
        TILE_SIZE = max(32, TILE_SIZE)

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
    NUM_BLOCKS_GATHER_PER_TILE: int = 1,
    SLIDING_WINDOW: int | None = None,
):
    arch = get_arch()
    reduce_num_warps = 2
    attn_warps = 2
    waves_per_eu = 2
    num_segments = 0
    attn_stages = 2
    if IS_DEVICE_ARCH_GFX12:
        assert kv_cache_dtype in (
            torch.float16,
            torch.bfloat16,
            e4m3_dtype,
            torch.uint8,
        ), (
            f"kv_cache_dtype only supports F16 ({torch.float16}) BF16 ({torch.bfloat16}), FP8 ({e4m3_dtype}), FP4 ({torch.uint8}) in arch = {DEVICE_ARCH}"
        )
        attn_warps = 1
        TILE_SIZE = block_size
        if shuffled_kv_cache and head_size < 128:
            if kv_cache_dtype in (
                torch.bfloat16,
                torch.float16,
            ):
                if block_size <= 64:
                    waves_per_eu = 2
                else:
                    waves_per_eu = 1
            elif kv_cache_dtype == e4m3_dtype:
                if block_size <= 128:
                    waves_per_eu = 2
                else:
                    waves_per_eu = 1
            else:
                assert block_size == 128, "FP4 KV cache only supports block_size 128"
                waves_per_eu = 2
        else:
            # GFX12 fallback
            waves_per_eu = 1

        if SLIDING_WINDOW is not None and SLIDING_WINDOW > 0:
            num_segments = 1
        else:
            occ = waves_per_eu * 4 // attn_warps
            MAX_SEGMENTS = max(1, math.ceil(max_seqlen_k / TILE_SIZE))
            num_segments = max(1, target_num_prgms // 4 * occ // max(1, num_2d_prgms))
            num_segments = min(MAX_SEGMENTS, num_segments)
            num_segments = triton.next_power_of_2(num_segments)

        # # this section increases the num_warps if the occ is too high
        # total_num_wg = num_2d_prgms * num_segments
        # if total_num_wg < occ * target_num_prgms:
        #     # occ too high, increase attn_warps to relax occ
        #     attn_warps = (waves_per_eu * 4) // max(
        #         1, triton.next_power_of_2(total_num_wg // target_num_prgms)
        #     )
        #     attn_warps = max(attn_warps, 1)
        #     attn_warps = min(attn_warps, 4)
    else:
        assert kv_cache_dtype in (
            torch.float16,
            torch.bfloat16,
            e4m3_dtype,
        ), (
            f"kv_cache_dtype only supports F16 ({torch.float16}) BF16 ({torch.bfloat16}), FP8 ({e4m3_dtype}) in arch = {DEVICE_ARCH}"
        )

        if head_size >= 512 and not arch.is_rdna:
            attn_warps, attn_stages = 4, 1
        occ = waves_per_eu * 4 // attn_warps
        target_num_prgms = target_num_prgms * occ

        TILE_SIZE = min(64, triton.next_power_of_2(block_size))

        MAX_SEGMENTS = min(128, math.ceil(max_seqlen_k / TILE_SIZE))
        MIN_SEGMENTS = min(8, MAX_SEGMENTS)
        if head_size >= 512 and not arch.is_rdna:
            MIN_SEGMENTS = min(16, MAX_SEGMENTS)
        if num_segments == 0:
            num_segments = math.ceil(target_num_prgms / num_2d_prgms)
            num_segments = min(num_segments, MAX_SEGMENTS)
            num_segments = max(num_segments, MIN_SEGMENTS)
            num_segments = triton.next_power_of_2(num_segments)

        if num_segments == MIN_SEGMENTS:
            reduce_num_warps = 1

        if shuffled_kv_cache:
            if q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:
                assert block_size >= 32, (
                    "For A8W8 Unified Attention with pre-shuffled KV cache, only block_size >= 32 is supported"
                )
            TILE_SIZE = block_size
        elif q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:
            TILE_SIZE = max(32, TILE_SIZE)

    if NUM_BLOCKS_GATHER_PER_TILE > 1:
        # force gather mode
        assert NUM_BLOCKS_GATHER_PER_TILE in [
            4,
            8,
        ], "Only NUM_BLOCKS_GATHER_PER_TILE = 4 or 8 is supported"
        attn_warps = 2
        waves_per_eu = 1
        num_segments = max(1, num_segments // NUM_BLOCKS_GATHER_PER_TILE)
        TILE_SIZE = block_size * NUM_BLOCKS_GATHER_PER_TILE
    elif TILE_SIZE > block_size:
        assert TILE_SIZE % block_size == 0, (
            "TILE_SIZE needs to be divisible by block_size"
        )
        NUM_BLOCKS_GATHER_PER_TILE = TILE_SIZE // block_size

    # gfx1151 (RDNA3.5) decode is memory-latency-bound at bs=1: the default 2
    # warps/workgroup leave unified_attention at only ~31% of the LPDDR5X
    # bandwidth roofline. 8 warps/workgroup reach ~59% (1.5-1.9x on bf16 decode)
    # with bitwise-identical output. Mirrors the waves_per_eu=8 gfx1151 tuning above.
    if DEVICE_ARCH == "gfx1151":
        attn_warps = 8

    attn_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": attn_warps,
        "waves_per_eu": waves_per_eu,
        "num_stages": attn_stages,
    }

    reduce_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": reduce_num_warps,
        "waves_per_eu": 2,
        "num_stages": 1,
    }

    return attn_config, reduce_config


def use_2d_kernel(params: _UAParams):
    # if IS_DEVICE_ARCH_GFX12, always use 3D if all_decode and 2D otherwise
    if IS_DEVICE_ARCH_GFX12:
        return (params.sliding_window > 0) or (not params.all_decode)

    if params.head_size >= 512 and not get_arch().is_rdna and not params.all_decode:
        return True

    return (
        (params.sliding_window > 0)
        or (params.max_seqlen_k <= 512)
        or (params.num_2d_prgms > params.target_num_prgms)
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
    q_scales=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    shuffled_kv_cache: bool = False,
    skip_reduce: bool = False,
):
    assert causal, "Only causal attention is supported"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None
    SLIDING_WINDOW = 1 + window_size[0]

    q_dtype = q.dtype
    kv_cache_dtype = k.dtype
    num_tokens, num_query_heads, head_size = q.shape

    if sinks is not None:
        assert sinks.shape[0] == num_query_heads, "Sinks must be num_query_heads size"

    BLOCK_SCALES_SIZE = 16
    if q_dtype == torch.uint8:
        # A4W4
        assert q_scales is not None and q_scales.dtype == e4m3_dtype
        head_size = head_size * 2
        QUERY_DTYPE = "nvfp4"
    elif q_dtype == e4m3_dtype:
        QUERY_DTYPE = "fp8"
    else:
        QUERY_DTYPE = "bf16"

    if kv_cache_dtype == torch.uint8:
        KV_CACHE_DTYPE = "nvfp4"
    elif kv_cache_dtype == e4m3_dtype:
        KV_CACHE_DTYPE = "fp8"
    else:
        KV_CACHE_DTYPE = "bf16"

    if shuffled_kv_cache:
        SCALE_K_WIDTH = 4
        if kv_cache_dtype == torch.uint8:
            num_blocks, num_kv_heads, block_size, _ = k.shape
            K_WIDTH = 16
            SCALE_K = head_size // 16
            SCALE_K_WIDTH = (
                min(16, triton.next_power_of_2(SCALE_K)) if SCALE_K >= 4 else SCALE_K
            )
        else:
            # key_cache: num_blocks, num_kv_heads, head_size // x, block_size, x
            # value_cache: num_blocks, num_kv_heads, block_size // x, head_size, x
            num_blocks, num_kv_heads, _, block_size, K_WIDTH = k.shape
    else:
        # key_cache and value_cache: num_blocks, block_size, num_kv_heads, head_size
        num_blocks, block_size, num_kv_heads, _ = k.shape
        K_WIDTH = 16 if kv_cache_dtype == e4m3_dtype else 8
        SCALE_K_WIDTH = 4

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
    cu_count = get_num_sms()
    target_num_prgms = cu_count * 4
    ALL_DECODE = max_seqlen_q == 1
    if ALL_DECODE:
        total_num_q_blocks = num_seqs
    else:
        total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    ALL_DECODE = int(max_seqlen_q) == 1

    # build parameters
    params = _UAParams(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        block_table=block_table,
        softmax_scale=softmax_scale,
        softcap=softcap,
        causal=causal,
        sliding_window=SLIDING_WINDOW,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_tokens=num_tokens,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        num_seqs=num_seqs,
        total_num_q_blocks=total_num_q_blocks,
        num_2d_prgms=num_2d_prgms,
        num_blocks=num_blocks,
        block_size=block_size,
        k_width=K_WIDTH,
        scale_k_width=SCALE_K_WIDTH,
        block_scales_size=BLOCK_SCALES_SIZE,
        q_dtype=q_dtype,
        kv_cache_dtype=kv_cache_dtype,
        all_decode=ALL_DECODE,
        shuffled_kv_cache=shuffled_kv_cache,
        use_alibi_slopes=use_alibi_slopes,
        use_qq_bias=use_qq_bias,
        num_sms=cu_count,
        target_num_prgms=target_num_prgms,
        sinks=sinks,
        alibi_slopes=alibi_slopes,
        qq_bias=qq_bias,
        q_scales=q_scales,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        output_scale=output_scale,
        skip_reduce=skip_reduce,
    )

    # if batch contains a prefill
    if use_2d_kernel(params):
        # The gfx1250 Gluon 2d kernel only handles bf16/fp8 q+kv (with optional
        # sinks / output_scale / shuffled_kv_cache)
        use_gluon_2d = is_2d_gluon_available(params)
        if use_gluon_2d:
            _unified_attention_2d_gfx1250(params)
        else:
            _unified_attention_2d_triton(params)
    else:
        NUM_BLOCKS_GATHER_PER_TILE = 1
        attn_config, reduce_config = select_3d_config(
            head_size,
            block_size,
            max_seqlen_k,
            target_num_prgms,
            num_2d_prgms,
            q_dtype,
            kv_cache_dtype,
            shuffled_kv_cache,
            NUM_BLOCKS_GATHER_PER_TILE,
            SLIDING_WINDOW,
        )
        NUM_SEGMENTS = attn_config["NUM_SEGMENTS_PER_SEQ"]

        if NUM_SEGMENTS > 1:
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
        else:
            segm_output = out
            segm_max = out  # dummy ptr
            segm_expsum = out  # dummy ptr

        use_gluon_3d = is_3d_gluon_available(params)
        if use_gluon_3d:
            _unified_attention_3d_gfx1250(
                params,
                BLOCK_M,
                BLOCK_Q,
                NUM_SEGMENTS,
                segm_output,
                segm_max,
                segm_expsum,
                attn_config,
                NUM_BLOCKS_GATHER_PER_TILE,
                QUERY_DTYPE,
                KV_CACHE_DTYPE,
            )
        else:
            _unified_attention_3d_triton(
                params,
                BLOCK_M,
                BLOCK_Q,
                NUM_SEGMENTS,
                segm_output,
                segm_max,
                segm_expsum,
                attn_config,
            )

        if NUM_SEGMENTS == 1:
            return segm_output
        elif skip_reduce:
            return segm_output, segm_max, segm_expsum

        head_size_padded = triton.next_power_of_2(head_size)
        # Gluon reduce (one workgroup/token, in-wave segment merge); valid for all-decode with small split counts, else the Triton reduce_segments.
        gluon_num_warps = 8 if num_query_heads % 8 == 0 else 4

        use_gluon_reduce = is_reduce_gluon_available(
            params, NUM_SEGMENTS, head_size_padded, gluon_num_warps
        )
        if use_gluon_reduce:
            _reduce_segments_gfx1250(
                params,
                NUM_SEGMENTS,
                segm_output,
                segm_max,
                segm_expsum,
                head_size_padded,
                gluon_num_warps,
                reduce_config,
            )
        else:
            _reduce_segments_triton(
                params,
                BLOCK_Q,
                segm_output,
                segm_max,
                segm_expsum,
                head_size_padded,
                reduce_config,
            )
    return out


def _unified_attention_2d_triton(params):
    config = select_2d_config(
        params.block_size,
        params.head_size,
        params.sliding_window,
        params.all_decode,
        params.max_seqlen_q,
        params.max_seqlen_k,
        params.num_queries_per_kv,
        params.num_2d_prgms,
        params.q_dtype,
        params.kv_cache_dtype,
        params.shuffled_kv_cache,
    )
    assert config["BLOCK_Q"] >= 1
    if params.all_decode:
        total_num_q_blocks = params.num_seqs
    else:
        total_num_q_blocks = params.num_tokens // config["BLOCK_Q"] + params.num_seqs

    kernel_unified_attention_2d[
        (
            params.num_kv_heads,
            total_num_q_blocks,
        )
    ](
        output_ptr=params.out,
        query_ptr=params.q,
        key_cache_ptr=params.k,
        value_cache_ptr=params.v,
        sink_ptr=params.sinks,
        block_tables_ptr=params.block_table,
        seq_lens_ptr=params.seqused_k,
        alibi_slopes_ptr=params.alibi_slopes,
        qq_bias_ptr=params.qq_bias,
        scale=params.softmax_scale,
        q_descale_ptr=params.q_descale,
        k_descale_ptr=params.k_descale,
        v_descale_ptr=params.v_descale,
        out_scale_ptr=params.output_scale,
        softcap=params.softcap,
        num_query_heads=params.num_query_heads,
        num_queries_per_kv=params.num_queries_per_kv,
        block_table_stride=params.block_table.stride(0),
        query_stride_0=params.q.stride(0),
        query_stride_1=params.q.stride(1),
        output_stride_0=params.out.stride(0),
        output_stride_1=params.out.stride(1),
        qq_bias_stride_0=params.qq_bias.stride(0) if params.use_qq_bias else 0,
        BLOCK_SIZE=params.block_size,
        HEAD_SIZE=params.head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(params.head_size),
        USE_ALIBI_SLOPES=params.use_alibi_slopes,
        USE_QQ_BIAS=params.use_qq_bias,
        USE_SOFTCAP=(params.softcap > 0),
        USE_SINKS=(params.sinks is not None),
        SLIDING_WINDOW=params.sliding_window,
        stride_k_cache_0=params.k.stride(0),
        stride_k_cache_1=params.k.stride(1),
        stride_k_cache_2=params.k.stride(2),
        stride_k_cache_3=params.k.stride(3),
        stride_v_cache_0=params.v.stride(0),
        stride_v_cache_1=params.v.stride(1),
        stride_v_cache_2=params.v.stride(2),
        stride_v_cache_3=params.v.stride(3),
        query_start_len_ptr=params.cu_seqlens_q,
        num_seqs=params.num_seqs,
        ALL_DECODE=params.all_decode,
        SHUFFLED_KV_CACHE=params.shuffled_kv_cache,
        K_WIDTH=params.k_width,
        **config,
    )


def _unified_attention_3d_triton(
    params,
    BLOCK_M,
    BLOCK_Q,
    NUM_SEGMENTS,
    segm_output,
    segm_max,
    segm_expsum,
    attn_config,
):
    kernel_unified_attention_3d[
        (params.total_num_q_blocks, params.num_kv_heads, NUM_SEGMENTS)
    ](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=params.q,
        key_cache_ptr=params.k,
        value_cache_ptr=params.v,
        sink_ptr=params.sinks,
        block_tables_ptr=params.block_table,
        seq_lens_ptr=params.seqused_k,
        alibi_slopes_ptr=params.alibi_slopes,
        qq_bias_ptr=params.qq_bias,
        scale=params.softmax_scale,
        q_descale_ptr=params.q_descale,
        k_descale_ptr=params.k_descale,
        v_descale_ptr=params.v_descale,
        out_scale_ptr=(
            params.output_scale
            if (params.output_scale is not None and NUM_SEGMENTS == 1)
            else None
        ),
        softcap=params.softcap,
        num_query_heads=params.num_query_heads,
        num_queries_per_kv=params.num_queries_per_kv,
        block_table_stride=params.block_table.stride(0),
        query_stride_0=params.q.stride(0),
        query_stride_1=params.q.stride(1),
        qq_bias_stride_0=params.qq_bias.stride(0) if params.use_qq_bias else 0,
        BLOCK_SIZE=params.block_size,
        HEAD_SIZE=params.head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(params.head_size),
        USE_ALIBI_SLOPES=params.use_alibi_slopes,
        USE_QQ_BIAS=params.use_qq_bias,
        USE_SOFTCAP=(params.softcap > 0),
        USE_SINKS=(params.sinks is not None),
        SLIDING_WINDOW=params.sliding_window,
        stride_k_cache_0=params.k.stride(0),
        stride_k_cache_1=params.k.stride(1),
        stride_k_cache_2=params.k.stride(2),
        stride_k_cache_3=params.k.stride(3),
        stride_v_cache_0=params.v.stride(0),
        stride_v_cache_1=params.v.stride(1),
        stride_v_cache_2=params.v.stride(2),
        stride_v_cache_3=params.v.stride(3),
        query_start_len_ptr=params.cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=params.num_seqs,
        BLOCK_M=BLOCK_M,
        ALL_DECODE=params.all_decode,
        SHUFFLED_KV_CACHE=params.shuffled_kv_cache,
        K_WIDTH=params.k_width,
        IS_Q_FP8=(params.q_dtype == e4m3_dtype),
        IS_KV_FP8=(params.kv_cache_dtype == e4m3_dtype),
        **attn_config,
    )


def _reduce_segments_triton(
    params,
    BLOCK_Q,
    segm_output,
    segm_max,
    segm_expsum,
    head_size_padded,
    reduce_config,
):
    reduce_segments[(params.num_tokens, params.num_query_heads)](
        output_ptr=params.out,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=params.seqused_k,
        num_seqs=params.num_seqs,
        num_query_heads=params.num_query_heads,
        out_scale_ptr=params.output_scale,
        output_stride_0=params.out.stride(0),
        output_stride_1=params.out.stride(1),
        block_table_stride=params.block_table.stride(0),
        HEAD_SIZE=params.head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        query_start_len_ptr=params.cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        **reduce_config,
    )


def _unified_attention_2d_gfx1250(params, loop_variant=None):
    """
    Internal wrapper for the gfx1250 gluon kernel.

    Args:
        params: shared parameters for this unified_attention call.
        loop_variant:
            0=plain double buffered version,
            1=2-stage version,
            2=4-stage version
    """
    # useful for debugging when needed
    remove_indirect_access = False
    NUM_SEQS = params.num_seqs
    NUM_Q_HEADS = params.num_query_heads
    HEAD_SIZE = params.head_size
    num_blocks = params.num_blocks
    Q_FP8 = params.q.element_size() == 1
    KV_FP8 = params.k.element_size() == 1
    ARCH_NAME = arch_info.get_arch()
    assert loop_variant in [
        None,
        0,
        1,
        2,
    ], "Only [None, 0, 1, 2] supported as loop_variant"
    assert ARCH_NAME == "gfx1250", "unified_attention_2d_gfx1250 only supports gfx1250"
    assert params.softcap == 0, "Softcap is not supported"
    BLOCK_SIZE = params.block_size
    NUM_KV_HEADS = params.num_kv_heads

    SLIDING_WINDOW = params.sliding_window
    ALL_DECODE = params.all_decode
    NUM_QUERIES_PER_KV = params.num_queries_per_kv
    num_buffers = None
    if ALL_DECODE:
        sel_loop_variant = 0
        BLOCK_M = (
            16
            if NUM_QUERIES_PER_KV <= 16
            else triton.next_power_of_2(NUM_QUERIES_PER_KV)
        )
        num_warps = 1
        waves_per_eu = 2
        TILE_SIZE = 128 if (Q_FP8 and KV_FP8) else 64
    elif SLIDING_WINDOW > 0:
        # Prefill, sliding window
        sel_loop_variant = 0
        BLOCK_M = 64
        num_warps = 4
        waves_per_eu = 4
        TILE_SIZE = 128 if (Q_FP8 and KV_FP8) else 64
    else:
        # Prefill, full attention
        sel_loop_variant = 2
        BLOCK_M = 128
        num_warps = 4
        waves_per_eu = 2
        TILE_SIZE = 128 if (Q_FP8 and KV_FP8) else 64
        num_buffers = 2

        if params.max_seqlen_k < 2048:
            BLOCK_M = 64
            num_warps = 2 if (Q_FP8 and KV_FP8) else 1
            sel_loop_variant = 0
            num_buffers = 2

    loop_variant = sel_loop_variant if loop_variant is None else loop_variant
    # Non-shuffled KV can't use TDM gather (KV layout), so a tile is one page
    if not params.shuffled_kv_cache or TILE_SIZE < BLOCK_SIZE:
        TILE_SIZE = BLOCK_SIZE

    num_kv_blocks = TILE_SIZE // BLOCK_SIZE if params.shuffled_kv_cache else 1
    assert num_kv_blocks & (num_kv_blocks - 1) == 0, (
        "num_kv_blocks must be a power of 2"
    )

    assert TILE_SIZE >= BLOCK_SIZE, (
        f"TILE_SIZE={TILE_SIZE} must be multiple of PAGE_SIZE={BLOCK_SIZE}"
    )

    BLOCK_Q = BLOCK_M // NUM_QUERIES_PER_KV
    # Upper bound on masked tiles
    query_span = (BLOCK_M - 1) // NUM_QUERIES_PER_KV + 1
    max_mask_tiles = (query_span + TILE_SIZE - 1) // TILE_SIZE + 1
    # other variants do at most 2 masking at the end of loop
    if max_mask_tiles > 2:
        loop_variant = 0
    # fall back to the standard double-buffered loop
    if TILE_SIZE <= 32:
        loop_variant = 0
    if not ALL_DECODE:
        total_query_blocks = params.num_tokens // BLOCK_Q + NUM_SEQS
    else:
        total_query_blocks = NUM_SEQS
    NUM_WARPS = num_warps
    if num_buffers is None:
        num_buffers = 2 if loop_variant == 0 else 3
        num_buffers = 2 if ALL_DECODE else num_buffers

    kv_size = params.k.nelement() * params.k.element_size()
    MAX_INT32 = 2**31 - 1
    USE_LOAD_BUFFER_OP = ARCH_NAME != "gfx1250" and kv_size <= MAX_INT32
    USE_STORE_BUFFER_OP = params.out.nelement() * params.out.element_size() <= MAX_INT32
    grid = (NUM_KV_HEADS, total_query_blocks)
    _unified_attention_gluon_kernel_2d[grid](
        query_ptr=params.q,
        key_cache_ptr=params.k,
        value_cache_ptr=params.v,
        sink_ptr=params.sinks,
        output_ptr=params.out,
        block_tables_ptr=params.block_table,
        seq_lens_ptr=params.seqused_k,
        query_start_len_ptr=params.cu_seqlens_q,
        query_stride_0=params.q.stride(0),
        query_stride_1=params.q.stride(1),
        output_stride_0=params.out.stride(0),
        output_stride_1=params.out.stride(1),
        k_descale_ptr=params.k_descale,
        v_descale_ptr=params.v_descale,
        q_descale_ptr=params.q_descale,
        out_scale_ptr=params.output_scale,
        USE_SINKS=(params.sinks is not None),
        SLIDING_WINDOW=SLIDING_WINDOW,
        num_blocks=num_blocks,
        stride_k_cache_0=params.k.stride(0),
        stride_k_cache_1=params.k.stride(1),
        stride_k_cache_2=params.k.stride(2),
        stride_k_cache_3=params.k.stride(3),
        stride_v_cache_0=params.v.stride(0),
        stride_v_cache_1=params.v.stride(1),
        stride_v_cache_2=params.v.stride(2),
        stride_v_cache_3=params.v.stride(3),
        block_table_stride=params.block_table.stride(0),
        num_seqs=NUM_SEQS,
        SCALE=params.softmax_scale,
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
        SHUFFLED_KV_CACHE=params.shuffled_kv_cache,
        CAUSAL=params.causal,
        REMOVE_INDIRECT_ACCESS=remove_indirect_access,
        NUM_BUFFERS=num_buffers,
        LOOP_VARIANT=loop_variant,
    )


def _unified_attention_3d_gfx1250(
    params,
    BLOCK_M,
    BLOCK_Q,
    NUM_SEGMENTS,
    segm_output,
    segm_max,
    segm_expsum,
    attn_config,
    NUM_BLOCKS_GATHER_PER_TILE,
    QUERY_DTYPE,
    KV_CACHE_DTYPE,
):
    _unified_attention_gluon_kernel_3d[
        (params.total_num_q_blocks, params.num_kv_heads, NUM_SEGMENTS)
    ](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=params.q,
        query_scales_ptr=params.q_scales,
        key_cache_ptr=params.k,
        value_cache_ptr=params.v,
        sink_ptr=params.sinks,
        block_tables_ptr=params.block_table,
        seq_lens_ptr=params.seqused_k,
        alibi_slopes_ptr=params.alibi_slopes,
        qq_bias_ptr=params.qq_bias,
        q_scale_ptr=params.q_descale,
        k_scale_ptr=params.k_descale,
        v_scale_ptr=params.v_descale,
        out_scale_ptr=(
            params.output_scale
            if (params.output_scale is not None and NUM_SEGMENTS == 1)
            else None
        ),
        softcap=params.softcap,
        num_seqs=params.num_seqs,
        num_blocks=params.num_blocks,
        block_table_stride=params.block_table.stride(0),
        max_num_blocks_per_seq=params.block_table.shape[1],
        query_stride_0=params.q.stride(0),
        query_stride_1=params.q.stride(1),
        query_scales_stride_0=params.q_scales.stride(0)
        if params.q_scales is not None
        else 0,
        query_scales_stride_1=params.q_scales.stride(1)
        if params.q_scales is not None
        else 0,
        qq_bias_stride_0=params.qq_bias.stride(0) if params.use_qq_bias else 0,
        BLOCK_SIZE=params.block_size,
        HEAD_SIZE=params.head_size,
        USE_ALIBI_SLOPES=params.use_alibi_slopes,
        USE_QQ_BIAS=params.use_qq_bias,
        USE_SOFTCAP=(params.softcap > 0),
        USE_SINKS=(params.sinks is not None),
        SLIDING_WINDOW=params.sliding_window,
        stride_k_cache_0=params.k.stride(0),
        stride_k_cache_1=params.k.stride(1),
        stride_k_cache_2=params.k.stride(2),
        stride_k_cache_3=params.k.stride(3),
        stride_v_cache_0=params.v.stride(0),
        stride_v_cache_1=params.v.stride(1),
        stride_v_cache_2=params.v.stride(2),
        stride_v_cache_3=params.v.stride(3),
        query_start_len_ptr=params.cu_seqlens_q,
        SCALE=params.softmax_scale,
        NUM_QUERY_HEADS=params.num_query_heads,
        NUM_KV_HEADS=params.num_kv_heads,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        ALL_DECODE=params.all_decode,
        SHUFFLED_KV_CACHE=params.shuffled_kv_cache,
        K_WIDTH=params.k_width,
        SCALE_K_WIDTH=params.scale_k_width,
        WARP_SIZE=WARP_SIZE,
        NUM_BLOCKS_GATHER_PER_TILE=NUM_BLOCKS_GATHER_PER_TILE,
        QUERY_DTYPE=QUERY_DTYPE,
        KV_CACHE_DTYPE=KV_CACHE_DTYPE,
        BLOCK_SCALES_SIZE=params.block_scales_size,
        **attn_config,
    )


def _reduce_segments_gfx1250(
    params,
    NUM_SEGMENTS,
    segm_output,
    segm_max,
    segm_expsum,
    head_size_padded,
    gluon_num_warps,
    reduce_config,
):
    _reduce_segments_gluon[(params.num_tokens,)](
        output_ptr=params.out,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=params.seqused_k,
        num_query_heads=params.num_query_heads,
        out_scale_ptr=params.output_scale,
        output_stride_0=params.out.stride(0),
        output_stride_1=params.out.stride(1),
        H=params.num_query_heads,
        S=NUM_SEGMENTS,
        D=params.head_size,
        D_PAD=head_size_padded,
        TILE_SIZE=reduce_config["TILE_SIZE"],
        NUM_WARPS=gluon_num_warps,
        IS_FP8_OUT=(params.out.dtype == e4m3_dtype),
        FP8_MIN=torch.finfo(e4m3_dtype).min,
        FP8_MAX=torch.finfo(e4m3_dtype).max,
        num_warps=gluon_num_warps,
    )
