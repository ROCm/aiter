# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
import json
import os
import triton
import torch
from typing import Optional
import triton.language as tl
from ..utils._triton.pid_preprocessing import pid_grid, remap_xcd
from ..utils._triton import arch_info
from ..utils.core import AITER_TRITON_CONFIGS_PATH
from ..utils._triton.kernel_repr import make_kernel_repr
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import os
from aiter.utility.triton.triton_metadata_redirect import AOTMetadataContext

global _USE_GEMM_SPLITK_BF16
_USE_GEMM_SPLITK_BF16 = False

@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
    }
)
@gluon.jit
def _gemm_afp4_wfp4_kernel_preshuffled_weight_scales(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    GRID_MN = gl.cdiv(M, BLOCK_SIZE_M) * gl.cdiv(N, BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = gl.program_id(axis=0)
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n


    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32

    blocked_ascale: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 2],
        threads_per_warp=[1, 64],
        warps_per_cta=[1, 2],
        order=[1, 0],
    )
    
    blocked_bscale: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[1, 64],
        warps_per_cta=[2, 1],
        order=[1, 0],
    )

    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[2, 1],
        order=[1, 0],
    )

    linear_nk: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 1024], [1, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [0, 512]],
        warp_bases=[[2, 0]],
        block_bases=[],
        shape=[BLOCK_SIZE_N // 16, BLOCK_SIZE_K // 2 * 16],
    )

    linear_ascale: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases = [[0, 2], [0, 1]],
        lane_bases = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]],
        warp_bases = [[0, 0]],
        block_bases = [],
        shape = [BLOCK_SIZE_M // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32],
    )

    linear_bscale: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases = [[0, 2], [0, 1]],
        lane_bases = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]],
        warp_bases = [[1, 0]],
        block_bases = [],
        shape= [BLOCK_SIZE_N // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32],
    )

    # linear1: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases = [[0, 2], [0, 1]],
    #     lane_bases = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]],
    #     warp_bases = [[1, 0]],
    #     block_bases = [],
    #     shape= [BLOCK_SIZE_N // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32],
    # )
    
    # linear2: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases = [[0, 2], [0, 1]],
    #     lane_bases = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]],
    #     warp_bases = [[1, 0]],
    #     block_bases = [],
    #     shape= [BLOCK_SIZE_N // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32],
    # )
    # linear3: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases = [[0, 2], [0, 1]],
    #     lane_bases = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]],
    #     warp_bases = [[1, 0]],
    #     block_bases = [],
    #     shape= [BLOCK_SIZE_N // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32],
    # )
    # linear4: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases = [[0, 2], [0, 1]],
    #     lane_bases = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]],
    #     warp_bases = [[1, 0]],
    #     block_bases = [],
    #     shape= [BLOCK_SIZE_N // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32],
    # )
    # linear6: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases = [[0, 2], [0, 1]],
    #     lane_bases = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]],
    #     warp_bases = [[1, 0]],
    #     block_bases = [],
    #     shape= [BLOCK_SIZE_N // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32],
    # )
    # linear7: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases = [[0, 2], [0, 1]],
    #     lane_bases = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]],
    #     warp_bases = [[1, 0]],
    #     block_bases = [],
    #     shape= [BLOCK_SIZE_N // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32],
    # )
    # linear8: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases = [[0, 2], [0, 1]],
    #     lane_bases = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]],
    #     warp_bases = [[1, 0]],
    #     block_bases = [],
    #     shape= [BLOCK_SIZE_N // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32],
    # )
    # linear9: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 8], [0, 0, 2, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
    #     lane_bases = [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 8, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0]],
    #     warp_bases = [[0, 2, 0, 0, 0, 0]],
    #     block_bases = [],
    #     shape= [1, 4, 4, 2, 16, 16],
    # )
    # linear10: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 8], [0, 0, 0, 2, 0, 0], [0, 1, 0, 0, 0, 0]],
    #     lane_bases = [[0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0], [0, 0, 4, 0, 0, 0], [0, 0, 8, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0]],
    #     warp_bases = [[0, 2, 0, 0, 0, 0]],
    #     block_bases = [],
    #     shape= [1, 4, 4, 2, 16, 16],
    # )
    # linear11: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 64], [16, 0]],
    #     lane_bases = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16], [0, 32]],
    #     warp_bases = [[32, 0]],
    #     block_bases = [],
    #     shape= [64, 128],
    # )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[1, 2],
        tiles_per_warp=[1, 2],
    )
   
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )

    shared_scales: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:

        num_k_iter = gl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

        offs_k = gl.arange(0, BLOCK_SIZE_K // 2, layout=gl.SliceLayout(0, blocked_mk))
        offs_k_shuffle_arr = gl.arange(0, (BLOCK_SIZE_K // 2) * 16, layout=gl.SliceLayout(0, linear_nk))
        offs_k_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k
        offs_k_shuffle = pid_k * (SPLITK_BLOCK_SIZE // 2) * 16 + offs_k_shuffle_arr

        offs_am = (pid_m * BLOCK_SIZE_M + 
                   gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk))) % M
        offs_bn = (pid_n * (BLOCK_SIZE_N // 16) + 
                   gl.arange(0, BLOCK_SIZE_N // 16, layout=gl.SliceLayout(1, linear_nk))) % N
        
        offs_a = offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        offs_b = offs_bn[:, None] * stride_bn + offs_k_shuffle[None, :] * stride_bk
        
        offs_bsn = (
            pid_n * (BLOCK_SIZE_N // 32) + gl.arange(0, (BLOCK_SIZE_N // 32), layout=gl.SliceLayout(1, blocked_bscale))
            ) % N
        offs_bsk = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE) * 32) + gl.arange(
            0, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32, layout=gl.SliceLayout(0, blocked_bscale)
        )
        # B scales are N x K even though B operand is K x N.
        offs_b_scale = offs_bsn[:, None] * stride_bsn + offs_bsk[None, :] * stride_bsk

        if BLOCK_SIZE_M < 32:
            offs_ks_non_shufl = (
                pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)
            ) + gl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE, layout=gl.SliceLayout(0, blocked_ascale))
            offs_a_scale = (offs_am[:, None] * stride_asm + offs_ks_non_shufl[None, :] * stride_ask)
        else:
            offs_asm = (
                pid_m * (BLOCK_SIZE_M // 32) + gl.arange(0, (BLOCK_SIZE_M // 32), layout=gl.SliceLayout(1, blocked_ascale))
            ) % M
            offs_ask = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE) * 32) + gl.arange(
                0, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32, layout=gl.SliceLayout(0, blocked_ascale)
            )            
            offs_a_scale = offs_asm[:, None] * stride_asm + offs_ask[None, :] * stride_ask

        smem_a = gl.allocate_shared_memory(
            a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K // 2], layout=shared_a
        )
        if BLOCK_SIZE_M < 32:
            smem_ascale = gl.allocate_shared_memory(
                a_scales_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE], layout=shared_scales
            )
        else:
            smem_ascale = gl.allocate_shared_memory(
                a_scales_ptr.type.element_ty, [BLOCK_SIZE_M // 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32], layout=shared_scales
            )

        smem_bscale = gl.allocate_shared_memory(
            b_scales_ptr.type.element_ty, [BLOCK_SIZE_N// 32, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32], layout=shared_scales
        )

        a_scales = gl.amd.cdna4.buffer_load(
            ptr=a_scales_ptr,
            offsets=offs_a_scale,
        )

        b_scales = gl.amd.cdna4.buffer_load(
            ptr=b_scales_ptr,
            offsets=offs_b_scale,
            cache=cache_modifier,
        )

        a = gl.amd.cdna4.buffer_load(
            ptr=a_ptr,
            offsets=offs_a,
        )

        b = gl.amd.cdna4.buffer_load(
            ptr=b_ptr,
            offsets=offs_b,
            cache=cache_modifier,
        )

        smem_ascale.store(a_scales)
        smem_bscale.store(b_scales)
        smem_a.store(a)
        
        accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout)

        cur_b = b
        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter - 1):
            # Advance the ptrs to the next K block.
            a_ptr += (BLOCK_SIZE_K // 2) * stride_ak
            b_ptr += (BLOCK_SIZE_K // 2) * 16 * stride_bk
            if BLOCK_SIZE_M < 32:
                a_scales_ptr += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
            else:
                a_scales_ptr += BLOCK_SIZE_K * stride_ask
            b_scales_ptr += BLOCK_SIZE_K * stride_bsk

            next_a_scales = gl.amd.cdna4.buffer_load(
                ptr=a_scales_ptr,
                offsets=offs_a_scale,
            )
            cur_a_scales = (
                smem_ascale.load(layout=linear_ascale)
                .reshape(
                    BLOCK_SIZE_M // 32,
                    BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                    4,
                    16,
                    2,
                    2,
                    1,)
                    .permute(0, 5, 3, 1, 4, 2, 6)
                    .reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
            
            next_b_scales = gl.amd.cdna4.buffer_load(
                ptr=b_scales_ptr,
                offsets=offs_b_scale,
                cache=cache_modifier,
            )
            cur_b_scales = (
                smem_bscale.load(layout=linear_bscale)
                .reshape(
                    BLOCK_SIZE_N // 32,
                    BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                    4,
                    16,
                    2,
                    2,
                    1,
                )
                .permute(0, 5, 3, 1, 4, 2, 6)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            )

            next_a = gl.amd.cdna4.buffer_load(
                ptr=a_ptr,
                offsets=offs_a,
            )
            cur_a = smem_a.load(layout=dot_a_layout)
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            
            next_b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
                cache=cache_modifier,
            )           
            cur_b = (
                    cur_b.reshape(
                        1,
                        BLOCK_SIZE_N // 16,
                        BLOCK_SIZE_K // 64,
                        2,
                        16,
                        16,
                    )
                    .permute(0, 1, 4, 2, 3, 5)
                    .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // 2)
                    .trans(1, 0)
                )
            cur_b = gl.convert_layout(value=cur_b, layout=dot_b_layout, assert_trivial=True)
            # cur_b = gl.convert_layout(cur_b, dot_b_layout, False)
            # cur_b = cur_b.reshape(1, BLOCK_SIZE_N // 16, BLOCK_SIZE_K // 64, 2, 16, 16)
            # cur_b = gl.convert_layout(
            #     value=cur_b,
            #     layout=linear9,
            # )
            # cur_b = gl.convert_layout(
            #     value=cur_b,
            #     layout=linear10,
            # )
            # cur_b = gl.convert_layout(
            #     value=cur_b,
            #     layout=linear11,
            # )
            # cur_b = gl.convert_layout(
            #     value=cur_b,
            #     layout=dot_b_layout,
            # )
            accumulator = gl.amd.cdna4.mfma_scaled(a=cur_a, a_scale=cur_a_scales, a_format="e2m1", b=cur_b, b_scale=cur_b_scales, b_format="e2m1", acc=accumulator)
            smem_ascale.store(next_a_scales)
            smem_bscale.store(next_b_scales)
            smem_a.store(next_a)
            cur_b = next_b

        
        cur_a_scales = (
            smem_ascale.load(layout=linear_ascale)
            .reshape(
                BLOCK_SIZE_M // 32,
                BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                4,
                16,
                2,
                2,
                1,)
                .permute(0, 5, 3, 1, 4, 2, 6)
                .reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            )
        
        cur_b_scales = (
                smem_bscale.load(layout=linear_bscale)
                .reshape(
                    BLOCK_SIZE_N // 32,
                    BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                    4,
                    16,
                    2,
                    2,
                    1,
                )
                .permute(0, 5, 3, 1, 4, 2, 6)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            )
        
        cur_a = smem_a.load(layout=dot_a_layout)
        cur_b = (
                cur_b.reshape(
                    1,
                    BLOCK_SIZE_N // 16,
                    BLOCK_SIZE_K // 64,
                    2,
                    16,
                    16,
                )
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // 2)
                .trans(1, 0)
            )
        cur_b = gl.convert_layout(value=cur_b, layout=dot_b_layout, assert_trivial=True)
        # cur_b = gl.convert_layout(
        #     value=cur_b,
        #     layout=linear9,
        # )
        # cur_b = gl.convert_layout(
        #     value=cur_b,
        #     layout=linear10,
        # )
        # cur_b = gl.convert_layout(
        #     value=cur_b,
        #     layout=linear11,
        # )
        # cur_b = gl.convert_layout(
        #     value=cur_b,
        #     layout=dot_b_layout,
        # )
        accumulator = gl.amd.cdna4.mfma_scaled(a=cur_a, a_scale=cur_a_scales, a_format="e2m1", b=cur_b, b_scale=cur_b_scales, b_format="e2m1", acc=accumulator)

        
        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)).to(gl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)).to(gl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        gl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
    shuffle: bool = False,
):
    shuffle_filename_suffix = "" if not shuffle else "_PRESHUFFLED"
    if not hasattr(_get_config, "_config_dict") or not hasattr(
        _get_config._config_dict, f"default{shuffle_filename_suffix}"
    ):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-AFP4WFP4{shuffle_filename_suffix}.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict[f"default{shuffle_filename_suffix}"] = config

    key = f"{N}_{K}{shuffle_filename_suffix}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-AFP4WFP4{shuffle_filename_suffix}-N={N}-K={2*K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = f"default{shuffle_filename_suffix}"  # fall back to default config

    if M < 32:
        BLK_M = triton.next_power_of_2(M)
        if BLK_M >= 16 and "small_M16" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["small_M16"]
        return _get_config._config_dict[key]["small"]
    elif M <= 128:
        BLK_M = triton.next_power_of_2(M)
        if BLK_M == 32:
            return _get_config._config_dict[key]["medium_M32"]
        elif BLK_M == 64:
            return _get_config._config_dict[key]["medium_M64"]
        elif BLK_M == 128:
            return _get_config._config_dict[key]["medium_M128"]
    elif M <= 256:
        return _get_config._config_dict[key]["large"]
    else:
        BLK_M = triton.next_power_of_2(M)
        if f"xlarge_M{BLK_M}" in _get_config._config_dict[key]:
            return _get_config._config_dict[key][f"xlarge_M{BLK_M}"]
        return _get_config._config_dict[key]["xlarge"]


def get_splitk(K: int, BLOCK_SIZE_K: int, NUM_KSPLIT: int):
    # heuristics for make "EVEN_K == True" as much as possible
    NUM_KSPLIT_STEP = 2
    BLOCK_SIZE_K_STEP = 2
    SPLITK_BLOCK_SIZE = (
        triton.cdiv((2 * triton.cdiv(K, NUM_KSPLIT)), BLOCK_SIZE_K) * BLOCK_SIZE_K
    )
    while NUM_KSPLIT > 1 and BLOCK_SIZE_K > 16:
        if (
            K % (SPLITK_BLOCK_SIZE // 2) == 0
            and SPLITK_BLOCK_SIZE % BLOCK_SIZE_K == 0
            and K % (BLOCK_SIZE_K // 2) == 0
        ):
            break
        elif K % (SPLITK_BLOCK_SIZE // 2) != 0 and NUM_KSPLIT > 1:
            NUM_KSPLIT = NUM_KSPLIT // NUM_KSPLIT_STEP
        elif SPLITK_BLOCK_SIZE % BLOCK_SIZE_K != 0:
            if NUM_KSPLIT > 1:
                NUM_KSPLIT = NUM_KSPLIT // NUM_KSPLIT_STEP
            elif BLOCK_SIZE_K > 16:
                BLOCK_SIZE_K = BLOCK_SIZE_K // BLOCK_SIZE_K_STEP
        elif K % (BLOCK_SIZE_K // 2) != 0 and BLOCK_SIZE_K > 16:
            BLOCK_SIZE_K = BLOCK_SIZE_K // BLOCK_SIZE_K_STEP
        else:
            break

        SPLITK_BLOCK_SIZE = (
            triton.cdiv((2 * triton.cdiv(K, NUM_KSPLIT)), BLOCK_SIZE_K) * BLOCK_SIZE_K
        )

    return SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT

def gemm_afp4wfp4_preshuffled_weight_scales(
    x,
    w,
    x_scales,
    w_scales,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    use_aot: Optional[bool] = True,
):
    """
    Computes matrix multiplication Y = X @ W^T with FP4 activations and FP4 weights using preshuffled weight scales.
    Weight matrix and scales are stored in optimized layout for improved performance.

    Args:
        x (torch.Tensor): FP4 E2M1 input matrix with shape (M, K).
        w (torch.Tensor): FP4 E2M1 weight matrix with shape (N//16, K*16), internally transposed.
            Preshuffled layout: logical shape after unpacking is (N, K).
        x_scales (torch.Tensor): E8M0 per-group scale for x with shape (M//32, K) if M >= 32,
            or (M, K//32) if M < 32.
        w_scales (torch.Tensor): E8M0 per-group scale for w with shape (N//32, K).
            Groups of 32 rows in N dimension share K scales.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).
        use_aot (Optional[bool]): Enable ahead-of-time compilation metadata.

    Returns:
        torch.Tensor: Output with shape (M, N).
    """

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

    M, K = x.shape
    N, K = w.shape
    N = N * 16
    K = K // 16

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K, True)

    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )

        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT

        if _USE_GEMM_SPLITK_BF16:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=y.dtype, device=y.device
            )
        else:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=y.device
            )
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        y_pp = None

    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K

    config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)
    if M < 32:
        assert (
            config["BLOCK_SIZE_M"] <= 16
        ), "for M < 32, BLOCK_SIZE_M must be 16 or less as x_scale are assumed to be un-shuffled"
    else:
        assert (
            config["BLOCK_SIZE_M"] >= 32
        ), "for M >= 32, BLOCK_SIZE_M must be 32 or more as x_scale are assumed to be preshuffled"

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )

    M_POW2 = triton.next_power_of_2(M)
    if M < 32 and M_POW2 > 16:
        M_POW2 = 16
    metadata_pth = f"{AITER_TRITON_CONFIGS_PATH}/gemm/aot/{_gemm_afp4_wfp4_kernel_preshuffled_weight_scales.fn.__name__}_M={M_POW2}-N={N}-K={K*2}"
    if use_aot and os.path.exists(metadata_pth):
        with AOTMetadataContext(
            _gemm_afp4_wfp4_kernel_preshuffled_weight_scales.fn.__name__,
            f"{metadata_pth}",
        ):
            _gemm_afp4_wfp4_kernel_preshuffled_weight_scales[grid](
                x,
                w,
                y if config["NUM_KSPLIT"] == 1 else y_pp,
                x_scales,
                w_scales,
                M,
                N,
                K,
                x.stride(0),
                x.stride(1),
                w.stride(0),
                w.stride(1),
                0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
                y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
                y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
                x_scales.stride(0),
                x_scales.stride(1),
                w_scales.stride(0),
                w_scales.stride(1),
                **config,
            )
    else:
        _gemm_afp4_wfp4_kernel_preshuffled_weight_scales[grid](
            x,
            w,
            y if config["NUM_KSPLIT"] == 1 else y_pp,
            x_scales,
            w_scales,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
            y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
            y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
            x_scales.stride(0),
            x_scales.stride(1),
            w_scales.stride(0),
            w_scales.stride(1),
            **config,
        )

    return y
