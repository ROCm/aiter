# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
from typing import NamedTuple

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    kernel_unified_attention_2d,
    kernel_unified_attention_3d,
    reduce_segments,
)
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import get_arch
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.utils.unified_attention_utils import (
    get_dtype_str,
    get_unified_attention_config,
)

# gfx1250
try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_3d import (
        _unified_attention_gluon_kernel_3d as _unified_attention_kernel_3d_gfx1250,
    )
except:  # noqa: E722
    _unified_attention_kernel_3d_gfx1250 = None
try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_2d import (
        _unified_attention_gluon_kernel_2d as _unified_attention_kernel_2d_gfx1250,
    )
except:  # noqa: E722
    _unified_attention_kernel_2d_gfx1250 = None
try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_reduce import (
        reduce_segments_gluon as _reduce_segments_kernel_gfx1250,
    )
except:  # noqa: E722
    _reduce_segments_kernel_gfx1250 = None

# Max NUM_SEGMENTS the gluon reduce holds in-thread; larger split counts fall back to the Triton reduce_segments.
_GLUON_REDUCE_MAX_SEGMENTS = 8

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


def _is_gluon_available():
    return any(supported in DEVICE_ARCH for supported in _GLUON_SUPPORTED_ARCHS)


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
    # backend
    backend: str | None = None,  # "triton" | "gluon"
):
    assert causal, "Only causal attention is supported"

    if backend is None:
        backend = "gluon" if _is_gluon_available() else "triton"
    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"
    if backend == "gluon":
        assert (
            _is_gluon_available()
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"

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
        assert q_scales is not None and q_scales.dtype == e4m3_dtype
        head_size = head_size * 2

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

    if shuffled_kv_cache:
        # A shuffled tile is exactly one page (the kernels index the block table
        # per tile and read TILE_SIZE * HEAD_SIZE_PADDED contiguous elements), so
        # TILE_SIZE is pinned to block_size and has to be a power of 2 for the
        # tl.arange over the tile. Non-shuffled pages have no such constraint:
        # there the block table is indexed per token, so a tile may straddle pages.
        assert block_size & (block_size - 1) == 0, (
            "Unified Attention with pre-shuffled KV cache requires a power-of-2 "
            f"page, got block_size={block_size}"
        )

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
    ALL_DECODE = int(max_seqlen_q) == 1
    if ALL_DECODE:
        total_num_q_blocks = num_seqs
    else:
        total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs
    num_2d_prgms = total_num_q_blocks * num_kv_heads

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
        use_gluon_2d = is_2d_gluon_available(params, backend)
        if use_gluon_2d:
            if DEVICE_ARCH == "gfx1250":
                _unified_attention_2d_gfx1250(params)
            else:
                assert False, f"No gluon subwrapper for {DEVICE_ARCH}"
        else:
            _unified_attention_2d_triton(params)
    else:
        config = get_unified_attention_config("kv_split", params, backend=backend)
        NUM_SEGMENTS = config["NUM_SEGMENTS"]
        if shuffled_kv_cache:
            TILE_SIZE = block_size
        else:
            TILE_SIZE = config["TILE_SIZE"]

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

        use_gluon_3d = is_3d_gluon_available(params, backend)
        if use_gluon_3d:
            if DEVICE_ARCH == "gfx1250":
                _unified_attention_3d_gfx1250(
                    params,
                    segm_output,
                    segm_max,
                    segm_expsum,
                    NUM_SEGMENTS,
                    TILE_SIZE,
                )
            else:
                assert False, f"No gluon subwrapper for {DEVICE_ARCH}"
        else:
            _unified_attention_3d_triton(
                params,
                segm_output,
                segm_max,
                segm_expsum,
                NUM_SEGMENTS,
                TILE_SIZE,
            )

        if NUM_SEGMENTS == 1:
            return segm_output
        elif skip_reduce:
            return segm_output, segm_max, segm_expsum

        # Gluon reduce (one workgroup/token, in-wave segment merge); valid for all-decode with small split counts, else the Triton reduce_segments.
        use_gluon_reduce = (
            is_reduce_gluon_available(params, NUM_SEGMENTS, backend)
            and backend == "gluon"
        )
        if use_gluon_reduce:
            if DEVICE_ARCH == "gfx1250":
                _reduce_segments_gfx1250(
                    params,
                    segm_output,
                    segm_max,
                    segm_expsum,
                    NUM_SEGMENTS,
                    TILE_SIZE,
                )
            else:
                assert False, f"No gluon subwrapper for {DEVICE_ARCH}"
        else:
            _reduce_segments_triton(
                params,
                segm_output,
                segm_max,
                segm_expsum,
                NUM_SEGMENTS,
                TILE_SIZE,
            )
    return out


def is_2d_gluon_available(params: _UAParams, backend: str):
    use_gluon = backend == "gluon" and _is_gluon_available()

    # arch-specific gates
    use_gluon_arch = False
    if DEVICE_ARCH == "gfx1250":
        use_gluon_arch = (
            _unified_attention_kernel_2d_gfx1250 is not None
            and not params.softcap
            and not params.use_qq_bias
            and not params.use_alibi_slopes
            and params.q_dtype != torch.uint8
            and params.kv_cache_dtype != torch.uint8
            and params.q_dtype == params.kv_cache_dtype
        )

    return use_gluon and use_gluon_arch


def is_3d_gluon_available(params: _UAParams, backend: str):
    use_gluon = backend == "gluon" and _is_gluon_available()

    # arch-specific gates
    use_gluon_arch = False
    if DEVICE_ARCH == "gfx1250":
        use_gluon_arch = (
            _unified_attention_kernel_3d_gfx1250 is not None
            and params.shuffled_kv_cache
        )

    return use_gluon and use_gluon_arch


def is_reduce_gluon_available(params: _UAParams, NUM_SEGMENTS, backend: str):
    use_gluon = backend == "gluon" and _is_gluon_available()

    head_size_padded = triton.next_power_of_2(params.head_size)
    gluon_num_warps = 8 if params.num_query_heads % 8 == 0 else 4

    # arch-specific gates
    use_gluon_arch = False
    if DEVICE_ARCH == "gfx1250":
        use_gluon_arch = (
            _reduce_segments_kernel_gfx1250 is not None
            and params.all_decode
            and NUM_SEGMENTS <= _GLUON_REDUCE_MAX_SEGMENTS
            and head_size_padded % 32 == 0
            and params.num_query_heads % gluon_num_warps == 0
        )

    return use_gluon and use_gluon_arch


def _unified_attention_2d_triton(params: _UAParams):
    if params.shuffled_kv_cache and (
        params.q_dtype == e4m3_dtype and params.kv_cache_dtype == e4m3_dtype
    ):
        assert (
            params.block_size >= 32
        ), "For A8W8 Unified Attention with pre-shuffled KV cache, only block_size >= 32 is supported"

    config = get_unified_attention_config("attn_2d", params, backend="triton")
    config["BLOCK_M"] = max(
        config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv)
    )
    config["BLOCK_Q"] = config["BLOCK_M"] // params.num_queries_per_kv
    assert config["BLOCK_Q"] >= 1
    if params.shuffled_kv_cache:
        config["TILE_SIZE"] = params.block_size
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
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
):
    config = get_unified_attention_config("attn_3d", params, backend="triton")
    config["BLOCK_M"] = max(
        config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv)
    )
    config["BLOCK_Q"] = config["BLOCK_M"] // params.num_queries_per_kv
    assert config["BLOCK_Q"] >= 1

    if params.all_decode:
        total_num_q_blocks = params.num_seqs
    else:
        total_num_q_blocks = params.num_tokens // config["BLOCK_Q"] + params.num_seqs

    kernel_unified_attention_3d[
        (total_num_q_blocks, params.num_kv_heads, NUM_SEGMENTS)
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
        num_seqs=params.num_seqs,
        ALL_DECODE=params.all_decode,
        SHUFFLED_KV_CACHE=params.shuffled_kv_cache,
        K_WIDTH=params.k_width,
        IS_Q_FP8=(params.q_dtype == e4m3_dtype),
        IS_KV_FP8=(params.kv_cache_dtype == e4m3_dtype),
        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
        TILE_SIZE=TILE_SIZE,
        **config,
    )


def _reduce_segments_triton(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
):
    head_size_padded = triton.next_power_of_2(params.head_size)
    config = get_unified_attention_config("reduce", params, backend="triton")

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
        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
        TILE_SIZE=TILE_SIZE,
        BLOCK_Q=None,
        **config,
    )


"""
Below are arch-specific Gluon kernel wrappers.
unified_attention() picks one per kernel through the is_*_gluon_available() gates,
so the wrappers for a given kernel follows the signature below:
```
_unified_attention_2d_{arch}(
    params: _UAParams,
): ...

_unified_attention_3d_{arch}(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
): ...

_reduce_segments_{arch}(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
): ...
```
"""


def _unified_attention_2d_gfx1250(params: _UAParams):
    """
    Internal wrapper for the gfx1250 gluon kernel.

    loop_variant:
        0=plain double buffered version,
        1=2-stage version,
        2=4-stage version
    """
    assert params.softcap == 0, "Softcap is not supported"

    config = get_unified_attention_config("attn_2d", params, backend="gluon")
    NUM_SEQS = params.num_seqs
    TILE_SIZE = config["TILE_SIZE"]
    BLOCK_M = max(config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv))
    loop_variant = config["LOOP_VARIANT"]

    # Non-shuffled KV can't use TDM gather (KV layout), so a tile is one page
    if not params.shuffled_kv_cache or TILE_SIZE < params.block_size:
        TILE_SIZE = params.block_size
    num_kv_blocks = TILE_SIZE // params.block_size
    assert (
        num_kv_blocks & (num_kv_blocks - 1) == 0
    ), f"TILE_SIZE={TILE_SIZE} must be a power-of-2 multiple of PAGE_SIZE={params.block_size}"

    # the loop variants other than 0 mask at most twice at the end of the loop,
    # and need a tile wider than 32
    query_span = (BLOCK_M - 1) // params.num_queries_per_kv + 1
    if query_span > TILE_SIZE or TILE_SIZE <= 32:
        loop_variant = 0

    BLOCK_Q = BLOCK_M // params.num_queries_per_kv
    assert BLOCK_Q >= 1
    if params.all_decode:
        total_query_blocks = NUM_SEQS
    else:
        total_query_blocks = params.num_tokens // BLOCK_Q + NUM_SEQS

    # buffer ops need the tensor to fit a 32-bit offset; gfx1250 loads through TDM
    MAX_INT32 = 2**31 - 1
    USE_STORE_BUFFER_OP = params.out.nelement() * params.out.element_size() <= MAX_INT32
    _unified_attention_kernel_2d_gfx1250[(params.num_kv_heads, total_query_blocks)](
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
        SLIDING_WINDOW=params.sliding_window,
        num_blocks=params.num_blocks,
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
        NUM_QUERY_HEADS=params.num_query_heads,
        NUM_KV_HEADS=params.num_kv_heads,
        BLOCK_SIZE=params.block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=params.head_size,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        ARCH_NAME=DEVICE_ARCH,
        waves_per_eu=config["waves_per_eu"],
        USE_LOAD_BUFFER_OP=False,
        USE_STORE_BUFFER_OP=USE_STORE_BUFFER_OP,
        num_warps=config["num_warps"],
        ALL_DECODE=params.all_decode,
        SHUFFLED_KV_CACHE=params.shuffled_kv_cache,
        CAUSAL=params.causal,
        # useful for debugging when needed
        REMOVE_INDIRECT_ACCESS=False,
        NUM_BUFFERS=config["NUM_BUFFERS"],
        LOOP_VARIANT=loop_variant,
    )


def _unified_attention_3d_gfx1250(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
):
    NUM_BLOCKS_GATHER_PER_TILE = 1
    QUERY_DTYPE = get_dtype_str(params.q_dtype)
    KV_CACHE_DTYPE = get_dtype_str(params.kv_cache_dtype)
    config = get_unified_attention_config("attn_3d", params, backend="gluon")
    config["BLOCK_M"] = max(
        config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv)
    )
    config["BLOCK_Q"] = config["BLOCK_M"] // params.num_queries_per_kv
    assert config["BLOCK_Q"] >= 1

    if params.all_decode:
        total_num_q_blocks = params.num_seqs
    else:
        total_num_q_blocks = params.num_tokens // config["BLOCK_Q"] + params.num_seqs

    _unified_attention_kernel_3d_gfx1250[
        (total_num_q_blocks, params.num_kv_heads, NUM_SEGMENTS)
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
        query_scales_stride_0=(
            params.q_scales.stride(0) if params.q_scales is not None else 0
        ),
        query_scales_stride_1=(
            params.q_scales.stride(1) if params.q_scales is not None else 0
        ),
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
        ALL_DECODE=params.all_decode,
        SHUFFLED_KV_CACHE=params.shuffled_kv_cache,
        K_WIDTH=params.k_width,
        SCALE_K_WIDTH=params.scale_k_width,
        WARP_SIZE=WARP_SIZE,
        NUM_BLOCKS_GATHER_PER_TILE=NUM_BLOCKS_GATHER_PER_TILE,
        QUERY_DTYPE=QUERY_DTYPE,
        KV_CACHE_DTYPE=KV_CACHE_DTYPE,
        BLOCK_SCALES_SIZE=params.block_scales_size,
        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
        TILE_SIZE=TILE_SIZE,
        **config,
    )


def _reduce_segments_gfx1250(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
):
    head_size_padded = triton.next_power_of_2(params.head_size)
    gluon_num_warps = 8 if params.num_query_heads % 8 == 0 else 4
    config = get_unified_attention_config("reduce", params, backend="gluon")

    _reduce_segments_kernel_gfx1250[(params.num_tokens,)](
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
        TILE_SIZE=TILE_SIZE,
        IS_FP8_OUT=(params.out.dtype == e4m3_dtype),
        FP8_MIN=torch.finfo(e4m3_dtype).min,
        FP8_MAX=torch.finfo(e4m3_dtype).max,
        NUM_WARPS=gluon_num_warps,
        num_warps=gluon_num_warps,
        **config,
    )
