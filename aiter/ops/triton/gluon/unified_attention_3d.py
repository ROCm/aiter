# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import triton
import torch
from aiter.ops.triton.utils.device_info import get_num_sms
import math
from aiter.ops.triton.gluon.unified_attention_3d_kernel import (
    gluon_kernel_unified_attention_3d,
    gluon_kernel_unified_attention_3d_async,
    gluon_reduce_segments,
)
from aiter.ops.triton.gluon.unified_attention_3d_kernel_tdm import (
    gluon_kernel_unified_attention_3d_tdm,
)
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import aiter.ops.triton.utils._triton.arch_info as arch_info

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64
WAPR_SIZE_LOG2 = int(math.log2(WARP_SIZE))
from aiter.ops.triton.utils.types import e4m3_dtype


def make_kv_cache_shuffled_layout(
    BLOCK_SIZE_N_SHFL,
    BLOCK_SIZE_INNER_DIM_SHFL,
    fastest_dim_num_warps,
    total_num_warps,
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
        + [[1 << v, 0] for v in range(num_warps_log2, BLOCK_SIZE_N_SHFL_log2)]
    )
    lane_bases = [
        [0, 1 << v]
        for v in range(coalesced_size_log2, coalesced_size_log2 + WAPR_SIZE_LOG2)
    ]
    if num_warps_log2 > 0:
        warp_bases = [[1 << v, 0] for v in range(0, num_warps_log2)]
    elif total_num_warps == 1:
        warp_bases = []
    else:
        warp_bases = [[0, 0]]

    layout = gl.constexpr(
        gl.DistributedLinearLayout(
            reg_bases=reg_bases,
            lane_bases=lane_bases,
            warp_bases=warp_bases,
            block_bases=[],
            shape=[BLOCK_SIZE_N_SHFL, BLOCK_SIZE_INNER_DIM_SHFL],
        )
    )
    return layout


def make_layout_3d(
    num_warps: int,
    BLOCK_M: int,
    TILE_SIZE: int,
    BLOCK_SIZE: int,
    NUM_BLOCKS_GATHER_PER_TILE: int,
    HEAD_SIZE_PADDED: int,
    shuffled_kv_cache: bool,
    q_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    use_tdm: bool,
    use_async: bool,
    use_swizzle: bool = False,
    use_gather: bool = False,
):

    if IS_DEVICE_ARCH_GFX12:
        QK_WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
            version=3,
            transposed=True,
            warp_bases=[(1 << i, 0) for i in range(int(math.log2(num_warps)))],
            reg_bases=[],
            instr_shape=[16, 16, 32 if q_dtype == torch.bfloat16 else 64],
        )

        PV_WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
            version=3,
            transposed=True,
            warp_bases=[(0, 1 << i) for i in range(int(math.log2(num_warps)))],
            reg_bases=[],
            instr_shape=[16, 16, 32 if kv_cache_dtype == torch.bfloat16 else 64],
        )
        Q_DOT_LAYOUT: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=QK_WMMA_LAYOUT, k_width=8
        )
        K_DOT_LAYOUT: gl.constexpr = gl.DotOperandLayout(
            operand_index=1, parent=QK_WMMA_LAYOUT, k_width=8
        )
        P_DOT_LAYOUT: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=PV_WMMA_LAYOUT, k_width=8
        )
        V_DOT_LAYOUT: gl.constexpr = gl.DotOperandLayout(
            operand_index=1, parent=PV_WMMA_LAYOUT, k_width=8
        )
    elif shuffled_kv_cache:
        QK_WMMA_LAYOUT: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[num_warps, 1] if use_async else [1, num_warps],
            # warps_per_cta=[1, num_warps],
            # warps_per_cta=[num_warps, 1],
        )
        PV_WMMA_LAYOUT: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 16 if TILE_SIZE <= 16 else 32],
            transposed=True,
            warps_per_cta=[1, num_warps],
        )
        Q_DOT_LAYOUT: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=QK_WMMA_LAYOUT, k_width=8
        )
        K_DOT_LAYOUT: gl.constexpr = gl.DotOperandLayout(
            operand_index=1, parent=QK_WMMA_LAYOUT, k_width=8
        )
        P_DOT_LAYOUT: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=PV_WMMA_LAYOUT, k_width=4 if TILE_SIZE <= 16 else 8
        )
        V_DOT_LAYOUT: gl.constexpr = gl.DotOperandLayout(
            operand_index=1, parent=PV_WMMA_LAYOUT, k_width=4 if TILE_SIZE <= 16 else 8
        )
    else:
        QK_WMMA_LAYOUT: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[num_warps, 1],
        )
        PV_WMMA_LAYOUT: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 16 if TILE_SIZE <= 16 else 32],
            transposed=True,
            warps_per_cta=[1, num_warps],
        )
        Q_DOT_LAYOUT: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=QK_WMMA_LAYOUT, k_width=8
        )
        K_DOT_LAYOUT: gl.constexpr = gl.DotOperandLayout(
            operand_index=1, parent=QK_WMMA_LAYOUT, k_width=8
        )
        P_DOT_LAYOUT: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=PV_WMMA_LAYOUT, k_width=4
        )
        V_DOT_LAYOUT: gl.constexpr = gl.DotOperandLayout(
            operand_index=1, parent=PV_WMMA_LAYOUT, k_width=4
        )

    if use_tdm or not use_swizzle:
        Q_SHARED_LAYOUT: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            interval_padding_pairs=[[HEAD_SIZE_PADDED, 8]],
            shape=[BLOCK_M, HEAD_SIZE_PADDED],
            order=[1, 0],
        )
        if use_gather:
            K_SHARED_LAYOUT: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
                interval_padding_pairs=[[BLOCK_SIZE * HEAD_SIZE_PADDED, 8]],
                shape=(
                    [NUM_BLOCKS_GATHER_PER_TILE, BLOCK_SIZE * HEAD_SIZE_PADDED]
                    if use_tdm
                    else [HEAD_SIZE_PADDED, TILE_SIZE]
                ),
                order=[1, 0],
            )
            V_SHARED_LAYOUT: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
                interval_padding_pairs=[[BLOCK_SIZE * HEAD_SIZE_PADDED, 8]],
                shape=[NUM_BLOCKS_GATHER_PER_TILE, BLOCK_SIZE * HEAD_SIZE_PADDED],
                order=[1, 0],
            )
        else:
            K_SHARED_LAYOUT: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
                interval_padding_pairs=[[HEAD_SIZE_PADDED, 8]],
                shape=(
                    [TILE_SIZE, HEAD_SIZE_PADDED]
                    if use_tdm
                    else [HEAD_SIZE_PADDED, TILE_SIZE]
                ),
                order=[1, 0],
            )
            V_SHARED_LAYOUT: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
                interval_padding_pairs=[[HEAD_SIZE_PADDED, 8]],
                shape=[TILE_SIZE, HEAD_SIZE_PADDED],
                order=[1, 0],
            )
    else:
        Q_SHARED_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(
            vec=8, per_phase=2, max_phase=8, order=[1, 0]
        )
        if shuffled_kv_cache:
            K_SHARED_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(
                vec=1, per_phase=1, max_phase=1, order=[1, 0]
            )
            V_SHARED_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(
                vec=1, per_phase=1, max_phase=1, order=[1, 0]
            )
        else:
            K_SHARED_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(
                vec=8, per_phase=2, max_phase=8, order=[0, 1]
            )
            V_SHARED_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(
                vec=1, per_phase=1, max_phase=1, order=[1, 0]
            )

    # size_per_thread along the fastest moving dimension is set to 8 (BF16)
    size_per_thread_fastest_dim = 8

    # size_per_thread * threads_per_warp along the fastest moving dimension is set to HEAD_SIZE_PADDED with only 1 warp_per_cta,
    # therefore, threads_per_warp along the fastest moving dimension should be HEAD_SIZE_PADDED // size_per_thread_fastest_dim
    # clamp the threads_per_warp along the fastest moving dimension to 1 ~ WARP_SIZE
    threads_per_warp_fastest_dim = max(
        min((HEAD_SIZE_PADDED // size_per_thread_fastest_dim), WARP_SIZE), 1
    )

    # in gfx950, ttg.async_copy_global_to_local will fail if threads_per_warp=[WARP_SIZE//4, 4] is used
    Q_LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, size_per_thread_fastest_dim],
        threads_per_warp=[
            WARP_SIZE // threads_per_warp_fastest_dim,
            threads_per_warp_fastest_dim,
        ],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    if shuffled_kv_cache:
        K_LOAD_LAYOUT = make_kv_cache_shuffled_layout(
            TILE_SIZE // 16,
            HEAD_SIZE_PADDED * 16,
            1 if (use_async or IS_DEVICE_ARCH_GFX12) else num_warps,
            num_warps,
        )
        V_LOAD_LAYOUT = make_kv_cache_shuffled_layout(
            HEAD_SIZE_PADDED // 16, TILE_SIZE * 16, num_warps, num_warps
        )
    else:
        K_LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[size_per_thread_fastest_dim, 1],
            threads_per_warp=[
                threads_per_warp_fastest_dim,
                WARP_SIZE // threads_per_warp_fastest_dim,
            ],
            warps_per_cta=[1, num_warps],
            order=[0, 1],
        )
        V_LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, size_per_thread_fastest_dim],
            threads_per_warp=[
                WARP_SIZE // threads_per_warp_fastest_dim,
                threads_per_warp_fastest_dim,
            ],
            warps_per_cta=[num_warps, 1],
            order=[1, 0],
        )

    # TODO: for future impl
    # ctas_per_cga = [1, 1]
    # cga_layout_Q = make_cga_layout(
    #     ctasPerCga=ctas_per_cga,
    #     ctaSplitNum=[ctas_per_cga[0], 1],
    #     ctaOrder=[0, 1]
    # )
    # cga_layout_K = make_cga_layout(
    #     ctasPerCga=ctas_per_cga,
    #     ctaSplitNum=[1, ctas_per_cga[1]],
    #     ctaOrder=[0, 1]
    # )
    # cga_layout_S = make_cga_layout(
    #     ctasPerCga=ctas_per_cga,
    #     ctaSplitNum=[ctas_per_cga[0], ctas_per_cga[1]],
    #     ctaOrder=[0, 1]
    # )

    return {
        "QK_WMMA_LAYOUT": QK_WMMA_LAYOUT,
        "PV_WMMA_LAYOUT": PV_WMMA_LAYOUT,
        "Q_DOT_LAYOUT": Q_DOT_LAYOUT,
        "K_DOT_LAYOUT": K_DOT_LAYOUT,
        "P_DOT_LAYOUT": P_DOT_LAYOUT,
        "V_DOT_LAYOUT": V_DOT_LAYOUT,
        "Q_SHARED_LAYOUT": Q_SHARED_LAYOUT,
        "K_SHARED_LAYOUT": K_SHARED_LAYOUT,
        "V_SHARED_LAYOUT": V_SHARED_LAYOUT,
        "Q_LOAD_LAYOUT": Q_LOAD_LAYOUT,
        "K_LOAD_LAYOUT": K_LOAD_LAYOUT,
        "V_LOAD_LAYOUT": V_LOAD_LAYOUT,
    }


def select_3d_config(
    head_size,
    block_size,
    element_size,
    max_seqlen_k,
    target_num_prgms,
    num_2d_prgms,
    BLOCK_M: int,
    HEAD_SIZE_PADDED: int,
    q_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    use_tdm: bool = False,
    num_tdm_gather: int = 1,
    use_async: bool = True,
    use_swizzle: bool = True,
    shuffled_kv_cache: bool = False,
):
    """
    if use_tdm is True, use_async and use_swizzle will be ignored
    if use_tdm is False, num_tdm_gather will be ignored
    if use_async is True, use_swizzle will be forced to True
    if use_tdm and use_async are False, num_stages will be ignored, use_swizzle determines whether to use PaddedSharedLayout or SwizzledSharedLayout
    """
    reduce_num_warps = 2
    attn_warps = 2
    if kv_cache_dtype == torch.bfloat16 and block_size == 64:
        attn_warps = 1
    if q_dtype == torch.bfloat16 and kv_cache_dtype == e4m3_dtype and block_size == 128:
        attn_warps = 1

    if shuffled_kv_cache:
        assert (
            block_size >= 64
        ), "Only block_size >= 64 is supported for shuffled KV cache"

    if use_tdm and num_tdm_gather > 1:
        assert num_tdm_gather in [4, 8], "num_tdm_gather must be 4 or 8"

    TILE_SIZE = block_size * num_tdm_gather

    MAX_SEGMENTS = min(128, math.ceil(max_seqlen_k / TILE_SIZE))
    num_segments = math.ceil(target_num_prgms / num_2d_prgms)
    num_segments = min(num_segments, MAX_SEGMENTS)
    num_segments = triton.next_power_of_2(num_segments)
    num_segments = min(num_segments, 128)
    MIN_SEGMENTS = 16 if TILE_SIZE <= 16 else 8
    num_segments = max(num_segments, MIN_SEGMENTS)

    # TODO: needs a better way to determine num_segments for TDM gather pipelined
    if use_tdm and num_tdm_gather > 1:
        num_segments = 4

    if num_segments == MIN_SEGMENTS:
        reduce_num_warps = 1

    config_parms = (
        attn_warps,
        BLOCK_M,
        TILE_SIZE,
        block_size,
        num_tdm_gather,
        HEAD_SIZE_PADDED,
        shuffled_kv_cache,
        q_dtype,
        kv_cache_dtype,
        use_tdm,
        use_async,
    )

    # num_tiles_per_seq = (max_seqlen_k // num_segments + TILE_SIZE - 1) // TILE_SIZE

    attn_stages = 1
    if use_tdm:
        # With TDM async_copy pipelined, use_swizzle will be ignored (padded smem layout is used always)
        attn_impl = gluon_kernel_unified_attention_3d_tdm
        layout_configs = {
            "NUM_BLOCKS_GATHER_PER_TILE": num_tdm_gather,
        }
        attn_stages = 2
    else:
        if use_async:
            # With async_copy pipelined, use_swizzle should always be True
            attn_impl = gluon_kernel_unified_attention_3d_async
            layout_configs = make_layout_3d(
                *config_parms,
                use_swizzle=True,
            )
            # gfx12 does not have async_copy.buffer_load_to_shared
            # TODO: check KV cache size to determine if use_buffer_load is needed in gfx950
            layout_configs["USE_LOAD_BUFFER_OP"] = not IS_DEVICE_ARCH_GFX12
            attn_stages = 2
        else:
            # Baseline kernel, num_stages does not matter, use_swizzle can be either True or False
            attn_impl = gluon_kernel_unified_attention_3d
            layout_configs = make_layout_3d(
                *config_parms,
                use_swizzle=use_swizzle,
            )
        layout_configs["TILE_SIZE"] = TILE_SIZE

    waves_per_eu = 2
    occ = waves_per_eu * 4 // attn_warps
    if (
        2 * attn_stages * TILE_SIZE * head_size * kv_cache_dtype.itemsize * occ
        >= 327680
    ):
        waves_per_eu = 0

    attn_config = {
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "WARP_SIZE": WARP_SIZE,
        "num_warps": attn_warps,
        "num_stages": attn_stages,
        "waves_per_eu": waves_per_eu,
        "IS_Q_FP8": (q_dtype == e4m3_dtype),
        "IS_KV_FP8": (kv_cache_dtype == e4m3_dtype),
        **layout_configs,
    }

    reduce_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": reduce_num_warps,
        "num_stages": 1,
        "waves_per_eu": 2,
    }

    return attn_config, reduce_config, attn_impl


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
    use_tdm: bool = False,
    num_tdm_gather: int = 1,
    use_async: bool = True,
    shuffled_kv_cache: bool = False,
):
    assert causal, "Only causal attention is supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None
    SLIDING_WINDOW = 1 + window_size[0]

    num_tokens, num_query_heads, head_size = q.shape
    q_dtype = q.dtype
    kv_cache_dtype = k.dtype
    if shuffled_kv_cache:
        # key_cache: num_blocks, num_kv_heads, block_size // 16, head_size * 16
        # value_cache: num_blocks, num_kv_heads, head_size // 16, block_size * 16
        num_blocks, num_kv_heads, block_size, _ = k.shape
        block_size = block_size * 16
    else:
        if use_tdm and num_tdm_gather > 1:
            # key_cache and value_cache: num_blocks, num_kv_heads, block_size, head_size
            num_blocks, num_kv_heads, block_size, _ = k.shape
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
    #    = floor(num_tokens / BLOCK_Q) + num_seqs
    cu_count = get_num_sms()
    ALL_DECODE = max_seqlen_q == 1
    if ALL_DECODE:
        total_num_q_blocks = num_seqs
    else:
        total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs
    target_num_prgms = cu_count * 4
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
        raise NotImplementedError("2D Gluon Unified Attention is not yet implemented.")
    else:
        head_size_padded = triton.next_power_of_2(head_size)
        assert head_size_padded == head_size, "head_size must be a power of 2"

        if not IS_DEVICE_ARCH_GFX12:
            assert use_tdm == False, "TDM is not supported on non-GFX12 devices"

        use_swizzle = None
        if use_tdm == True:  # TDM
            use_async = None
        elif use_async == True:  # ASYNC
            pass
        else:  # Baseline (use_swizzle can be either True or False, fix to True for now)
            use_swizzle = True

        attn_config, reduce_config, attn_impl = select_3d_config(
            head_size,
            block_size,
            q.element_size(),
            max_seqlen_k,
            target_num_prgms,
            num_2d_prgms,
            BLOCK_M,
            head_size_padded,
            q_dtype,
            kv_cache_dtype,
            use_tdm,
            num_tdm_gather,
            use_async,
            use_swizzle,
            shuffled_kv_cache,
        )
        NUM_SEGMENTS = attn_config["NUM_SEGMENTS_PER_SEQ"]
        segm_output = torch.empty(
            num_tokens,
            num_query_heads,
            NUM_SEGMENTS,
            triton.next_power_of_2(head_size),
            dtype=torch.float32,
            device=q.device,
        )
        segm_max = torch.empty(
            num_tokens,
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
        segm_expsum = torch.empty(
            num_tokens,
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )

        attn_impl[(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
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
            **attn_config,
        )

        gluon_reduce_segments[(q.shape[0], num_query_heads)](
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
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            USE_FP8=output_scale is not None,
            **reduce_config,
        )
