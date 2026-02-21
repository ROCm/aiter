# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import triton
import torch
from aiter.ops.triton.utils.device_info import get_num_sms
import math
from aiter.ops.triton.gluon.unified_attention_3d_kernel import (
    gluon_kernel_unified_attention_3d,
    gluon_kernel_unified_attention_3d_async,
    gluon_kernel_unified_attention_3d_tdm,
    gluon_kernel_unified_attention_3d_tdm_gather,
    gluon_reduce_segments,
)
from aiter.ops.triton.gluon.unified_attention_3d_kernel_tdm_new import (
    kernel_unified_attention_3d_new,
)
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
import aiter.ops.triton.utils._triton.arch_info as arch_info
import numpy as np

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64


def make_distributed_linear_layout(
    shape: list[int], bases_representation: dict[str, list[str]]
):
    # rules:
    #   1. shape = [X, Y]
    #   2. x_log_2 = log2(X), y_log_2 = log2(Y)
    #   3. there are a total of x_log_2 + y_log_2 tokens: "x0", "x1", ... "x{x_log_2 - 1}", "y0", "y1", ... "y{x_log_2 - 1}"
    #   4. these tokens must appear exactly once in the bases_representation across "reg_bases", "lane_bases", "warp_bases" keys:
    #
    # example: shape = [64, 128]
    #   available tokens: "x0", "x1", "x2", "x3", "x4", "x5", "y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7"
    #   each of the above tokens must appear once in the bases_representation across "reg_bases", "lane_bases", "warp_bases" keys
    #
    # note:
    #    although this will not cause any compailation error outside gluon.jit,
    #    you need to make sure "lane_bases" has exactly 6/5 tokens for gfx950/gfx1250, respectively, because there are 64/32 threads per warp
    #    also you need to make sure "warp_bases" has exactly log2(num_warps) tokens, e.g., 1 token for num_warps = 2, 2 tokens for num_warps = 4, etc.

    token_counter = {}
    for i, d in [(0, "x"), (1, "y")]:
        for lg in range(int(math.log2(shape[i]))):
            token_counter[f"{d}{lg}"] = 0

    arg = {"reg_bases": [], "lane_bases": [], "warp_bases": []}

    for k in arg.keys():
        for v in bases_representation[k]:
            if v not in token_counter:
                raise ValueError(
                    f"Token {v} is invalid, it should be one of {token_counter.keys()}"
                )
            token_counter[v] += 1
            d = v[0]
            lg = int(v[1])
            if d == "x":
                arg[k].append([0, 2**lg])
            else:
                arg[k].append([2**lg, 0])

    for k in token_counter.keys():
        if token_counter[k] != 1:
            raise ValueError(
                f"Token {k} should appear only once, but appears {token_counter[k]} times"
            )
    # print(arg)
    layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        **arg,
        block_bases=[],
        shape=shape,
    )
    return layout


def get_layout_view(layout, shape):
    layout_view = layout.format_tensor_view(shape)
    layout_array = []
    for l in layout_view.strip("[]").strip("\n").split("\n"):
        layout_array.append([])
        for ll in l.strip("[]").split(","):
            t = int(ll.strip(" ").split(":")[0][1:])
            e = int(ll.strip(" ").split(":")[-1])
            layout_array[-1].append([t, e])
    return layout_view, torch.tensor(layout_array).to(torch.int32)


def get_smem_data_layout(layout, shape, bases_representation):
    _, layout_array = get_layout_view(layout.format_tensor_view(shape))
    contiguous_dim = bases_representation["reg_bases"][0][0]
    assert bases_representation["reg_bases"][0][1] == "0"
    for log_coalesced in range(1, len(bases_representation["reg_bases"])):
        if (
            bases_representation["reg_bases"][log_coalesced]
            != f"{contiguous_dim}{2**(log_coalesced - 1)}"
        ):
            break
    coalesced_size = 2**log_coalesced
    smem_data_layout = []
    for tid in range(WARP_SIZE):
        smem_data_layout.append([])
        data_idx = (layout_array[:, :, 0] == tid).nonzero().T
        element_idx = layout_array[data_idx[0], data_idx[1], 1]
        assert (
            len(element_idx) % coalesced_size == 0
        ), f"There must be {coalesced_size} elements alinged per thread"
        num_vmem_load = len(element_idx) // coalesced_size
        for i_vmem_load in range(num_vmem_load):
            assert torch.allclose(
                element_idx[
                    i_vmem_load * coalesced_size : (i_vmem_load + 1) * coalesced_size
                ]
                - element_idx[i_vmem_load * coalesced_size],
                torch.arange(coalesced_size, dtype=torch.int32),
            ), f"There must be {coalesced_size} contiguous elements per vmem load"
        i_vmem_load_order = torch.argsort(element_idx[::coalesced_size])
        for i_vmem_load in range(num_vmem_load):
            idx_row = data_idx[0][
                i_vmem_load_order[i_vmem_load]
                * coalesced_size : (i_vmem_load_order[i_vmem_load] + 1)
                * coalesced_size
            ]
            idx_col = data_idx[1][
                i_vmem_load_order[i_vmem_load]
                * coalesced_size : (i_vmem_load_order[i_vmem_load] + 1)
                * coalesced_size
            ]
            smem_data_layout[-1].append(torch.stack((idx_row, idx_col)).T)
        smem_data_layout[-1] = torch.stack(smem_data_layout[-1])
    smem_data_layout = torch.stack(smem_data_layout)
    # [num_vmem_loads, num_threads, coalesced_size, 2]
    smem_data_layout = smem_data_layout.permute(1, 0, 2, 3)


def make_layout_3d(
    num_warps: int,
    BLOCK_M: int,
    TILE_SIZE: int,
    BLOCK_SIZE: int,
    NUM_BLOCKS_GATHER_PER_TILE: int,
    HEAD_SIZE_PADDED: int,
    use_tdm: bool,
    use_swizzle: bool = False,
    use_gather: bool = False,
):

    if IS_DEVICE_ARCH_GFX12:
        # BLOCK_M are usually 16 (QH per KVH are usually <= 16)
        # TILE_SIZE are usually 16 or 64
        # HEAD_SIZE_PADDED are usually 64 or 128

        # for Q @ K^T (M x N x K = BLOCK_M x TILE_SIZE x HEAD_SIZE_PADDED),
        # the M-dim can usually be completed by 1 wave, while N-dim requires multiple waves and/or cycles,
        # so the best choice for warp_bases is:
        #     [[0, 1]]         for num_warps = 2, and

        #         w0 w1 ...
        #         ...

        #     [[0, 1], [0, 2]] for num_warps = 4

        #         w0 w1 w2 w3 ...
        #         ...

        # for P @ V (M x N x K = BLOCK_M x HEAD_SIZE_PADDED x TILE_SIZE),
        # the M-dim can usually be completed by 1 wave, while N-dim requires multiple waves and/or cycles,
        # so the best choice for warp_bases is the same as Q @ K^T

        # some examples for warp_bases for num_warps = 4

        #     warp_bases=[[0, 1], [1, 0]]
        #     w0 w1 ...
        #     w2 w3 ...
        #     ...

        #     warp_bases=[[1, 0], [0, 1]]
        #     w0 w2 ...
        #     w1 w3 ...
        #     ...

        #     warp_bases=[[0, 1], [0, 2]]
        #     w0 w1 w2 w3 ...
        #     ...

        # therefore, we construct WMMA layout with the following heuristics

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
    else:
        QK_WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[2, num_warps // 2],
        )
        PV_WMMA_LAYOUT: ttgl.constexpr = ttgl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 16 if TILE_SIZE <= 16 else 32],
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
            operand_index=0, parent=PV_WMMA_LAYOUT, k_width=4
        )
        V_DOT_LAYOUT: ttgl.constexpr = ttgl.DotOperandLayout(
            operand_index=1, parent=PV_WMMA_LAYOUT, k_width=4
        )

    if use_tdm or not use_swizzle:
        Q_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
            interval_padding_pairs=[[HEAD_SIZE_PADDED, 8]],
            shape=[BLOCK_M, HEAD_SIZE_PADDED],
            order=[1, 0],
        )
        if use_gather:
            K_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
                interval_padding_pairs=[[BLOCK_SIZE * HEAD_SIZE_PADDED, 8]],
                shape=(
                    [NUM_BLOCKS_GATHER_PER_TILE, BLOCK_SIZE * HEAD_SIZE_PADDED]
                    if use_tdm
                    else [HEAD_SIZE_PADDED, TILE_SIZE]
                ),
                order=[1, 0],
            )
            V_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
                interval_padding_pairs=[[BLOCK_SIZE * HEAD_SIZE_PADDED, 8]],
                shape=[NUM_BLOCKS_GATHER_PER_TILE, BLOCK_SIZE * HEAD_SIZE_PADDED],
                order=[1, 0],
            )
        else:
            K_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
                interval_padding_pairs=[[HEAD_SIZE_PADDED, 8]],
                shape=(
                    [TILE_SIZE, HEAD_SIZE_PADDED]
                    if use_tdm
                    else [HEAD_SIZE_PADDED, TILE_SIZE]
                ),
                order=[1, 0],
            )
            V_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for(
                interval_padding_pairs=[[HEAD_SIZE_PADDED, 8]],
                shape=[TILE_SIZE, HEAD_SIZE_PADDED],
                order=[1, 0],
            )
    else:
        Q_SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(
            vec=8, per_phase=2, max_phase=8, order=[1, 0]
        )
        # K_SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(
        #     vec=8, per_phase=2, max_phase=8, order=[0, 1]
        # )
        # V_SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(
        #     vec=1, per_phase=1, max_phase=1, order=[1, 0]
        # )
        # V_SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(
        #     vec=4, per_phase=2, max_phase=8, order=[1, 0]
        # )

        # 64
        # ttg.padded_shared<[512:+16] {offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16], [0, 1], [0, 2], [0, 32]], block = []}>
        # 128
        # ttg.padded_shared<[512:+16] {offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 1], [0, 2], [0, 4], [0, 8]], block = []}>
        K_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [32, 0],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 1],
                [0, 2],
                [0, 32],
            ],
            cga_layout=[],
            shape=[HEAD_SIZE_PADDED, TILE_SIZE],
        )

        # 64
        # ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [8, 0], [4, 0], [16, 0], [1, 0], [2, 0], [32, 0]], block = []}>
        # 128
        # ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [1, 0], [2, 0], [8, 0], [4, 0]], block = []}>
        V_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [8, 0],
                [4, 0],
                [16, 0],
                [1, 0],
                [2, 0],
                [32, 0],
            ],
            cga_layout=[],
            shape=[TILE_SIZE, HEAD_SIZE_PADDED],
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
    Q_BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(
        size_per_thread=[1, size_per_thread_fastest_dim],
        threads_per_warp=[
            WARP_SIZE // threads_per_warp_fastest_dim,
            threads_per_warp_fastest_dim,
        ],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    # K_BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(
    #     size_per_thread=[size_per_thread_fastest_dim, 1],
    #     threads_per_warp=[
    #         threads_per_warp_fastest_dim,
    #         WARP_SIZE // threads_per_warp_fastest_dim,
    #     ],
    #     warps_per_cta=[1, num_warps],
    #     order=[0, 1],
    # )
    # V_BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout(
    #     size_per_thread=[1, size_per_thread_fastest_dim],
    #     threads_per_warp=[
    #         WARP_SIZE // threads_per_warp_fastest_dim,
    #         threads_per_warp_fastest_dim,
    #     ],
    #     warps_per_cta=[num_warps, 1],
    #     order=[1, 0],
    # )

    # 64
    # ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 2], [0, 32]], lane = [[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]], warp = [[0, 1]], block = []}>
    # 128
    # ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 2], [0, 4], [0, 8]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]], warp = [[0, 1]], block = []}>
    # K_BLOCKED_LAYOUT: ttgl.constexpr = ttgl.DistributedLinearLayout(
    #     reg_bases=[[1, 0], [2, 0], [4, 0], [0, 2], [0, 32]],
    #     lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
    #     warp_bases=[[0, 1]],
    #     block_bases=[],
    #     shape=[HEAD_SIZE_PADDED, TILE_SIZE],
    # )
    bases_representation = {
        "reg_bases": ["y0", "y1", "y2", "x1", "x5"],
        "lane_bases": ["y3", "y4", "y5", "x2", "x3", "x4"],
        "warp_bases": ["x0"],
    }
    K_BLOCKED_LAYOUT = make_distributed_linear_layout(
        [HEAD_SIZE_PADDED, TILE_SIZE], bases_representation
    )

    # 64
    # ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [2, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [8, 0], [4, 0], [16, 0]], warp = [[1, 0]], block = []}>
    # 128
    # ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [2, 0], [8, 0], [4, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]], warp = [[1, 0]], block = []}>
    # V_BLOCKED_LAYOUT: ttgl.constexpr = ttgl.DistributedLinearLayout(
    #     reg_bases=[[0, 1], [0, 2], [0, 4], [2, 0], [32, 0]],
    #     lane_bases=[[0, 8], [0, 16], [0, 32], [8, 0], [4, 0], [16, 0]],
    #     warp_bases=[[1, 0]],
    #     block_bases=[],
    #     shape=[TILE_SIZE, HEAD_SIZE_PADDED],
    # )
    bases_representation = {
        "reg_bases": ["x0", "x1", "x2", "y1", "y5"],
        "lane_bases": ["x3", "x4", "x5", "y3", "y2", "y4"],
        "warp_bases": ["y0"],
    }
    V_BLOCKED_LAYOUT = make_distributed_linear_layout(
        [TILE_SIZE, HEAD_SIZE_PADDED], bases_representation
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
        "Q_BLOCKED_LAYOUT": Q_BLOCKED_LAYOUT,
        "K_BLOCKED_LAYOUT": K_BLOCKED_LAYOUT,
        "V_BLOCKED_LAYOUT": V_BLOCKED_LAYOUT,
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
    num_tdm_gather: int = 1,
    use_async: bool = True,
    use_swizzle: bool = True,
):
    """
    if use_tdm is True, use_async and use_swizzle will be ignored
    if use_tdm is False, num_tdm_gather will be ignored
    if use_async is True, use_swizzle will be forced to True
    if use_tdm and use_async are False, num_stages will be ignored, use_swizzle determines whether to use PaddedSharedLayout or SwizzledSharedLayout
    """
    reduce_num_warps = 2
    attn_warps = 2

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

    hyper_parms = (
        attn_warps,
        BLOCK_M,
        TILE_SIZE,
        block_size,
        num_tdm_gather,
        HEAD_SIZE_PADDED,
    )

    # num_tiles_per_seq = (max_seqlen_k // num_segments + TILE_SIZE - 1) // TILE_SIZE

    attn_stages = 1
    if use_tdm:
        # With TDM async_copy pipelined, use_swizzle will be ignored (padded smem layout is used always)
        if num_tdm_gather > 1:
            attn_impl = gluon_kernel_unified_attention_3d_tdm_gather
            # print(f"Using TDM gather pipelined kernel with TILE_SIZE={TILE_SIZE} and BLOCK_SIZE={block_size}")
            layouts = make_layout_3d(
                *hyper_parms, use_tdm, use_swizzle=False, use_gather=True
            )
        else:
            attn_impl = gluon_kernel_unified_attention_3d_tdm
            # print(f"Using TDM async copy pipelined kernel with TILE_SIZE={TILE_SIZE} and BLOCK_SIZE={block_size}")
            layouts = make_layout_3d(*hyper_parms, use_tdm, use_swizzle=False)
        attn_stages = 2
    elif use_async:
        # With async_copy pipelined, use_swizzle should always be True
        attn_impl = gluon_kernel_unified_attention_3d_async
        layouts = make_layout_3d(*hyper_parms, use_tdm, use_swizzle=True)
        # gfx12 does not have async_copy.buffer_load_to_shared
        # TODO: check KV cache size to determine if use_buffer_load is needed in gfx950
        layouts["use_buffer_load"] = not IS_DEVICE_ARCH_GFX12
        attn_stages = 2
    else:
        # Baseline kernel, num_stages does not matter, use_swizzle can be either True or False
        attn_impl = gluon_kernel_unified_attention_3d
        layouts = make_layout_3d(
            *hyper_parms,
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
    use_tdm: bool = False,
    num_tdm_gather: int = 1,
    use_async: bool = True,
    shuffled_kv_cache: bool = False,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None
    SLIDING_WINDOW = 1 + window_size[0]

    num_tokens, num_query_heads, head_size = q.shape
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
    total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs
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
            use_tdm,
            num_tdm_gather,
            use_async,
            use_swizzle,
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
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            softcap=softcap,
            num_tokens=num_tokens,
            NUM_BLOCKS=num_blocks,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
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
        # kernel_unified_attention_3d_new[
        #     (total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)
        # ](
        #     segm_output_ptr=segm_output,
        #     segm_max_ptr=segm_max,
        #     segm_expsum_ptr=segm_expsum,
        #     query_ptr=q,
        #     key_cache_ptr=k,
        #     value_cache_ptr=v,
        #     sink_ptr=sinks,
        #     block_tables_ptr=block_table,
        #     seq_lens_ptr=seqused_k,
        #     alibi_slopes_ptr=alibi_slopes,
        #     qq_bias_ptr=qq_bias,
        #     k_scale=k_descale,
        #     v_scale=v_descale,
        #     softcap=softcap,
        #     num_seqs=num_seqs,
        #     num_blocks=num_blocks,
        #     query_stride_0=q.stride(0),
        #     query_stride_1=q.stride(1),
        #     qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        #     USE_ALIBI_SLOPES=use_alibi_slopes,
        #     USE_QQ_BIAS=use_qq_bias,
        #     USE_SOFTCAP=(softcap > 0),
        #     USE_SINKS=(sinks is not None),
        #     SLIDING_WINDOW=SLIDING_WINDOW,
        #     stride_k_cache_0=k.stride(0),
        #     stride_k_cache_1=k.stride(1),
        #     stride_k_cache_2=k.stride(2),
        #     stride_k_cache_3=k.stride(3),
        #     stride_v_cache_0=v.stride(0),
        #     stride_v_cache_1=v.stride(1),
        #     stride_v_cache_2=v.stride(2),
        #     stride_v_cache_3=v.stride(3),
        #     block_table_stride=block_table.stride(0),
        #     query_start_len_ptr=cu_seqlens_q,
        #     SCALE=softmax_scale,
        #     NUM_QUERY_HEADS=num_query_heads,
        #     NUM_KV_HEADS=num_kv_heads,
        #     BLOCK_SIZE=block_size,
        #     HEAD_SIZE=head_size,
        #     BLOCK_Q=BLOCK_Q,
        #     BLOCK_M=BLOCK_M,
        #     NUM_SEGMENTS_PER_SEQ=attn_config["NUM_SEGMENTS_PER_SEQ"],
        #     WARP_SIZE=32,
        #     num_warps=attn_config["num_warps"],
        #     waves_per_eu=attn_config["waves_per_eu"],
        #     NUM_STAGES=attn_config["num_stages"],
        #     NUM_BLOCKS_GATHER_PER_TILE=attn_config["TILE_SIZE"] // block_size,
        #     ALL_DECODE=ALL_DECODE,
        # )

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
