# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import triton
import torch
from aiter.ops.triton.utils.device_info import get_num_sms
import math
from aiter.ops.triton._triton_kernels.attention.unified_attention_gluon import (
    gluon_kernel_unified_attention_3d,
    gluon_kernel_unified_attention_3d_pipelined,
    gluon_kernel_unified_attention_3d_tdm_pipelined,
    gluon_reduce_segments,
)

from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
import aiter.ops.triton.utils._triton.arch_info as arch_info

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64


def make_layout_3d(
    num_warps: int,
    BLOCK_M: int,
    TILE_SIZE: int,
    HEAD_SIZE_PADDED: int,
    use_tdm: bool,
    use_swizzle: bool = False,
):
    """
    BLOCK_M are usually 16 (QH per KVH are usually <= 16)
    TILE_SIZE are usually 16 or 64
    HEAD_SIZE_PADDED are usually 64 or 128

    for Q @ K^T (M x N x K = BLOCK_M x TILE_SIZE x HEAD_SIZE_PADDED),
    the M-dim can usually be completed by 1 wave, while N-dim requires multiple waves and/or cycles,
    so the best choice for warp_bases is:
        [[0, 1]]         for num_warps = 2, and

            w0 w1 ...
            ...

        [[0, 1], [0, 2]] for num_warps = 4

            w0 w1 w2 w3 ...
            ...

    for P @ V (M x N x K = BLOCK_M x HEAD_SIZE_PADDED x TILE_SIZE),
    the M-dim can usually be completed by 1 wave, while N-dim requires multiple waves and/or cycles,
    so the best choice for warp_bases is the same as Q @ K^T

    some examples for warp_bases for num_warps = 4

        warp_bases=[[0, 1], [1, 0]]
        w0 w1 ...
        w2 w3 ...
        ...

        warp_bases=[[1, 0], [0, 1]]
        w0 w2 ...
        w1 w3 ...
        ...

        warp_bases=[[0, 1], [0, 2]]
        w0 w1 w2 w3 ...
        ...

    therefore, we construct WMMA layout with the following heuristics
    """

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

    if IS_DEVICE_ARCH_GFX12:
        warp_bases = [(0, 1 << i) for i in range(int(math.log2(num_warps)))]

        QK_WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(
            version=3,
            transposed=True,
            warp_bases=warp_bases,
            reg_bases=[],
            instr_shape=[16, 16, 32],
        )

        PV_WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDWMMALayout(
            version=3,
            transposed=True,
            warp_bases=warp_bases,
            reg_bases=[],
            instr_shape=[16, 16, 32],
        )
    else:
        QK_WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[num_warps // 2, 2],
        )
        PV_WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 16],
            transposed=True,
            warps_per_cta=[num_warps // 2, 2],
        )

    Q_DOT_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(
        operand_index=0, parent=QK_WMMA_LAYOUT, k_width=8
    )
    K_DOT_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(
        operand_index=1, parent=QK_WMMA_LAYOUT, k_width=8
    )
    P_DOT_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(
        operand_index=0, parent=PV_WMMA_LAYOUT, k_width=8
    )
    V_DOT_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(
        operand_index=1, parent=PV_WMMA_LAYOUT, k_width=8
    )

    if use_tdm or not use_swizzle:
        Q_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
            interval_padding_pairs=[[HEAD_SIZE_PADDED, 8]],
            shape=[BLOCK_M, HEAD_SIZE_PADDED],
            order=[1, 0],
        )
    else:
        Q_SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(
            vec=8, per_phase=1, max_phase=8, order=[1, 0]
        )

    if use_tdm or not use_swizzle:
        K_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
            interval_padding_pairs=[[HEAD_SIZE_PADDED, 8]],
            shape=(
                [TILE_SIZE, HEAD_SIZE_PADDED]
                if use_tdm
                else [HEAD_SIZE_PADDED, TILE_SIZE]
            ),
            order=[1, 0],
        )
    else:
        K_SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(
            vec=8, per_phase=1, max_phase=8, order=[0, 1]
        )

    if use_tdm or not use_swizzle:
        V_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
            interval_padding_pairs=[[HEAD_SIZE_PADDED, 8]],
            shape=[TILE_SIZE, HEAD_SIZE_PADDED],
            order=[1, 0],
        )
    else:
        V_SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[1, 0]
        )

    Q_BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[
            WARP_SIZE // 8,
            8,
        ],  # in gfx950, ttg.async_copy_global_to_local will fail if threads_per_warp=[WARP_SIZE//4, 4] is used
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    K_BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[8, WARP_SIZE // 8],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )

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
        "Q_BLOCKED_LAYOUT": Q_BLOCKED_LAYOUT,
        "K_BLOCKED_LAYOUT": K_BLOCKED_LAYOUT,
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
    use_tdm: bool = False,
    use_async: bool = True,
    use_swizzle: bool = True,
):
    """
    if use_tdm is True, use_async and use_swizzle will be ignored
    if use_async is True, use_swizzle will be forced to True
    if use_tdm and use_async are False, num_stages will be ignored, use_swizzle determines whether to use PaddedSharedLayout or SwizzledSharedLayout
    """
    reduce_num_warps = 2
    attn_warps = 2
    # attn_warps = 4
    TILE_SIZE = block_size
    MAX_SEGMENTS = min(128, math.ceil(max_seqlen_k / TILE_SIZE))
    num_segments = math.ceil(target_num_prgms / num_2d_prgms)
    num_segments = triton.next_power_of_2(num_segments)
    num_segments = min(num_segments, 128)
    MIN_SEGMENTS = 16 if TILE_SIZE <= 16 else 8
    num_segments = max(num_segments, MIN_SEGMENTS)

    attn_stages = 2 if num_segments > 1 else 1

    if num_segments == MIN_SEGMENTS:
        reduce_num_warps = 1

    if use_tdm:
        # With TDM async_copy pipelined, use_swizzle will be ignored (padded smem layout is used always)
        attn_impl = gluon_kernel_unified_attention_3d_tdm_pipelined
        layouts = make_layout_3d(
            attn_warps, BLOCK_M, TILE_SIZE, HEAD_SIZE_PADDED, use_tdm, use_swizzle=False
        )
    elif use_async:
        # With async_copy pipelined, use_swizzle should always be True
        attn_impl = gluon_kernel_unified_attention_3d_pipelined
        layouts = make_layout_3d(
            attn_warps, BLOCK_M, TILE_SIZE, HEAD_SIZE_PADDED, use_tdm, use_swizzle=True
        )
    else:
        # Baseline kernel, num_stages does not matter, use_swizzle can be either True or False
        attn_impl = gluon_kernel_unified_attention_3d
        layouts = make_layout_3d(
            attn_warps,
            BLOCK_M,
            TILE_SIZE,
            HEAD_SIZE_PADDED,
            use_tdm,
            use_swizzle=use_swizzle,
        )

    attn_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": attn_warps,
        "num_stages": attn_stages,
        "waves_per_eu": 2,
        **layouts,
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
    ver=0,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None
    SLIDING_WINDOW = 1 + window_size[0]

    num_tokens = q.shape[0]
    num_blocks = v.shape[0]
    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

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
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    target_num_prgms = cu_count * 4
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    ALL_DECODE = max_seqlen_q == 1
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

        if not IS_DEVICE_ARCH_GFX12:
            assert ver < 3

        if ver == 3:
            # TDM:
            use_tdm = True
            use_async = None
            use_swizzle = None
        elif ver == 2:
            # ASYNC
            use_tdm = False
            use_async = True
            use_swizzle = None
        elif ver == 1:
            # Baseline swizzle
            use_tdm = False
            use_async = False
            use_swizzle = True
        elif ver == 0:
            # Baseline pad
            use_tdm = False
            use_async = False
            use_swizzle = False

        attn_config, reduce_config, attn_impl = select_3d_config(
            head_size,
            block_size,
            q.element_size(),
            max_seqlen_k,
            target_num_prgms,
            num_2d_prgms,
            BLOCK_M,
            head_size_padded,
            use_tdm,
            use_async,
            use_swizzle,
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

        # for parm, val in attn_config.items():
        #     print(parm, val)
        # print(attn_impl.__name__)
        # print(attn_config["Q_SHARED_LAYOUT"])
        # print(attn_config["K_SHARED_LAYOUT"])
        # print(attn_config["V_SHARED_LAYOUT"])

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
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            softcap=softcap,
            num_tokens=num_tokens,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            NUM_BLOCKS=num_blocks,
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
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
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            USE_FP8=output_scale is not None,
            **reduce_config,
        )
