# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
import math
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.logger import AiterTritonLogger
from triton import language as tl

_LOGGER = AiterTritonLogger()
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def create_tensor_descriptors(a_ptr, b_ptr, off_am, off_bn, stride_am, stride_ak, stride_bn, stride_bk,
                              shared_layout_a: gl.constexpr, shared_layout_b: gl.constexpr, M,
                              N, K, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                              BLOCK_K: gl.constexpr, TRANSPOSE_B: gl.constexpr):

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr + off_am, shape=(M, K),
                                                         strides=(stride_am, stride_ak), block_shape=(BLOCK_M, BLOCK_K),
                                                         layout=shared_layout_a)
    if not TRANSPOSE_B:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr + off_bn, shape=(K, N),
                                                             strides=(stride_bk, stride_bn),
                                                             block_shape=(BLOCK_K, BLOCK_N), layout=shared_layout_b)
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr + off_bn, shape=(N, K),
                                                             strides=(stride_bn, stride_bk),
                                                             block_shape=(BLOCK_N, BLOCK_K), layout=shared_layout_b)

    return a_desc, b_desc

@gluon.jit
def issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K: gl.constexpr,
                NUM_BUFFERS: gl.constexpr, TRANSPOSE_B: gl.constexpr, pred=1):
    # pred is a hardware predicate passed to async_load for conditional execution without branch divergence
    # Convert boolean pred to i32 for hardware predicate (i1 -> i32)
    pred_i32 = pred.to(gl.int32) if hasattr(pred, 'to') else pred
    gl.amd.gfx1250.tdm.async_load(a_desc, [off_am, producer * BLOCK_K], a_buffer.index(producer % NUM_BUFFERS),
                                    pred=1)
    if not TRANSPOSE_B:
        gl.amd.gfx1250.tdm.async_load(b_desc, [producer * BLOCK_K, off_bn], b_buffer.index(producer % NUM_BUFFERS),
                                        pred=1)
    else:
        gl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, producer * BLOCK_K], b_buffer.index(producer % NUM_BUFFERS),
                                        pred=1)
    producer += 1
    return producer

@gluon.jit
def issue_l2_prefetches(distance, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K: gl.constexpr,
                        TRANSPOSE_B: gl.constexpr, pred=True):
    """
    Creates L2 prefetch for iteration `producer + distance`.
    """
    if distance == 0:
        return

    prefetch_iteration = producer + distance
    gl.amd.gfx1250.tdm.prefetch(a_desc, [off_am, prefetch_iteration * BLOCK_K], pred=pred)
    if not TRANSPOSE_B:
        gl.amd.gfx1250.tdm.prefetch(b_desc, [prefetch_iteration * BLOCK_K, off_bn], pred=pred)
    else:
        gl.amd.gfx1250.tdm.prefetch(b_desc, [off_bn, prefetch_iteration * BLOCK_K], pred=pred)


@gluon.jit
def issue_l2_prefetches_prologue(distance, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K: gl.constexpr,
                                 NUM_BUFFERS: gl.constexpr, TRANSPOSE_B: gl.constexpr, pred=True):
    """
    Creates prefetches for iterations [NUM_BUFFERS, distance - NUM_BUFFERS) or no prefetches if distance <= NUM_BUFFERS.
    This skips iterations which are preloaded in the prologue because prefetching them does not make sense for GEMMs.
    """
    if distance <= NUM_BUFFERS:
        return

    for i in gl.static_range(NUM_BUFFERS - distance):
        issue_l2_prefetches(distance + NUM_BUFFERS + i, producer, a_desc, b_desc, 0, 0, BLOCK_K, TRANSPOSE_B, pred)



@gluon.jit
def issue_wmma(consumer, a_buffer, a_layout: gl.constexpr, b_buffer, b_layout: gl.constexpr, accumulator,
               wait_producers_cnt, NUM_BUFFERS: gl.constexpr, TRANSPOSE_B: gl.constexpr):
    """
    For multi-CTA configurations, we want warps within the CGA (cluster) to stay temporally aligned so we can
    multicast data to multiple CTAs.
    We do this by signaling the cluster barrier before `async_wait` (which inserts a CTA barrier), then waiting
    for the cluster barrier to complete. This keeps warps of a CGA within one iteration of each other.
    It can also improve latency hiding by overlapping the cluster and CTA barriers.
    """
    num_ctas: gl.constexpr = gl.num_ctas()
    if num_ctas > 1:
        gl.amd.gfx1250.cluster.arrive()

    gl.amd.gfx1250.tdm.async_wait(wait_producers_cnt)

    if num_ctas > 1:
        gl.amd.gfx1250.cluster.wait()

    a = a_buffer.index(consumer % NUM_BUFFERS).load(layout=a_layout)
    if not TRANSPOSE_B:
        b = b_buffer.index(consumer % NUM_BUFFERS).load(layout=b_layout)
    else:
        b = b_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=b_layout)

    accumulator = gl.amd.gfx1250.wmma(a, b, accumulator)
    consumer += 1
    return consumer, accumulator


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit
def _gemm_a8w8_blockscale_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: gl.constexpr,
    GROUP_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    warp_bases: gl.constexpr,
    cache_modifier: gl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8_blockscale function
    below

    Computes the 8 bit matmul C = A x B using the block-scale quantization approach.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N).
    - A_scale: Scale tensor for A with shape (M, *scale_k).
    - B_scale: Scale tensor for B with shape (*scale_k, **scale_n).

    *scale_k = (K + GROUP_K - 1) // GROUP_K
    **scale_n = (N + GROUP_N - 1) // GROUP_N
    """

    # TDM stats
    NUM_BUFFERS: gl.constexpr = 2
    L2_PREFETCH_DISTANCE: gl.constexpr = 1

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = gl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    # NOTE: there is no instruction i can see in the shader guide for int8 or fp8 wmma so i've put a temporary cast.
    # if i find a better way to do this that would be better, but should be noted

    # Setup
    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    threads_per_elem_mk: gl.constexpr = 1#triton.cdiv(
    #     BLOCK_SIZE_M * BLOCK_SIZE_K // (NUM_WARPS * 64), 16
    # )
    threads_per_elem_kn: gl.constexpr = 1#triton.cdiv(
    #     BLOCK_SIZE_K * BLOCK_SIZE_N // (NUM_WARPS * 64), 16
    # )
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[threads_per_elem_mk, 16],
        threads_per_warp=[4, 8], # 32 bc its 32 threads in a warp
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, threads_per_elem_kn],
        threads_per_warp=[8, 4], # 32
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )

    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(3, True, warp_bases, [], [16, 16, 64])


    # TDM Shared Layouts
    tdm_shared_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_SIZE_K, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_K],
                                                                                [1, 0])
    tdm_shared_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_SIZE_N, 8]], [BLOCK_SIZE_K, BLOCK_SIZE_N],
                                                                                    [0, 1])
    
    # unswizzled scales
    shared_a_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    shared_b_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=8
    )

    if (pid_k * SPLITK_BLOCK_SIZE) < K:
        #SPLITK_BLOCK_SIZE = gl.cdiv(K, NUM_KSPLIT)
        num_k_iter = gl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        # Create pointers for first block of A and B input matrices
        offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_mk))
        offs_ak_split = pid_k * SPLITK_BLOCK_SIZE + offs_ak
        offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_kn))
        offs_bk_split = pid_k * SPLITK_BLOCK_SIZE + offs_bk

        smem_scale_a = gl.allocate_shared_memory(
            gl.float32, [BLOCK_SIZE_M], layout=shared_a_scale
        )

        smem_scale_b = gl.allocate_shared_memory(
            gl.float32, [BLOCK_SIZE_N], layout=shared_b_scale
        )

        offs_am = pid_m * BLOCK_SIZE_M + gl.arange(
            0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk)
        )
        offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(
            0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn)
        )

        offs_a = offs_am[:, None] * stride_am + offs_ak_split[None, :] * stride_ak

        # Create pointers for the scales
        offs_k_scale = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        offs_a_scale = offs_am * stride_ascale_m + offs_k_scale * stride_ascale_k
        
        
        # TDM tensor descriptors and shared mem
        a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, 0, 0,
                                               stride_am, stride_ak, stride_bn, stride_bk, tdm_shared_a,
                                               tdm_shared_b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, False)
        tdm_smem_a = gl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
        tdm_smem_b = gl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

        # TODO change these variable names to something not producer/consumer. leaving as is for now for clarity
        producer = 0 # producer
        consumer = 0 # consumer

        # Load A scale from global
        a_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=offs_a_scale,
            cache=cache_modifier,
        )

        offs_b = offs_bk_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        offs_b_scale_n = offs_bn // GROUP_N
        offs_b_scale = offs_k_scale * stride_bscale_k + offs_b_scale_n * stride_bscale_n
        # Load b scale from global
        b_scale = gl.amd.cdna4.buffer_load(
            ptr=b_scale_ptr,
            offsets=offs_b_scale,
            cache=cache_modifier,
        )
        # prefetch lds
        issue_l2_prefetches_prologue(L2_PREFETCH_DISTANCE, producer, a_desc, b_desc, 0, 0, BLOCK_SIZE_K, NUM_BUFFERS, False)

        off_am_tdm: gl.int32 = pid_m * BLOCK_SIZE_M
        off_bm_tdm: gl.int32 = pid_n * BLOCK_SIZE_N
        # Loading initial batch of a and b to be in the queue. now 2 things in queue (with num buffers 2)
        for _ in gl.static_range(NUM_BUFFERS - 1):
            producer = issue_loads(producer, a_desc, b_desc, off_am_tdm, off_bm_tdm, tdm_smem_a, tdm_smem_b, BLOCK_SIZE_K, NUM_BUFFERS, False)
       
        smem_scale_a.store(a_scale)
        smem_scale_b.store(b_scale) 

        acc_dtype = gl.float32 if c_ptr.type.element_ty != gl.int8 else gl.int32
        acc = gl.zeros(
            (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout
        )
        zeros = gl.zeros(
            (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout
        )

        offs_ks_step = BLOCK_SIZE_K // GROUP_K  # could be replaced by a constant 1

        for k in range(pid_k * num_k_iter, ((pid_k + 1) * num_k_iter) - 1):
            # Advance the ptrs to the next K block.
            a_scale_ptr += offs_ks_step * stride_ascale_k
            b_scale_ptr += offs_ks_step * stride_bscale_k

            # Loading a scale and curr A scale
            cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
            a_scale = gl.amd.cdna4.buffer_load(
                ptr=a_scale_ptr,
                offsets=offs_a_scale,
                cache=cache_modifier,
            )
            # Loading b scale and curr b scale
            cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))
            b_scale = gl.amd.cdna4.buffer_load(
                ptr=b_scale_ptr,
                offsets=offs_b_scale,
                cache=cache_modifier,
            )

            
            # Load A for tiling. now loads both A and B
            producer = issue_loads(producer, a_desc, b_desc, off_am_tdm, off_bm_tdm, tdm_smem_a, tdm_smem_b, BLOCK_SIZE_K, NUM_BUFFERS, False)
            
            # prefetching
            issue_l2_prefetches(L2_PREFETCH_DISTANCE - 1, producer, a_desc, b_desc, 0, 0, BLOCK_SIZE_K, False)

            # WMMA but the legit way. Clear zeros afterwards.
            consumer, zeros = issue_wmma(consumer, tdm_smem_a, dot_a_layout, tdm_smem_b, dot_b_layout, zeros, 2, NUM_BUFFERS, False)
            acc += zeros * cur_a_scale[:, None] * cur_b_scale[None, :]
            zeros = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)

            smem_scale_a.store(a_scale)
            smem_scale_b.store(b_scale) 

        # ======= Epilogue ========

        # load a from the last load
        cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
        cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))

        for i in gl.static_range(NUM_BUFFERS - 1):
            consumer, zeros = issue_wmma(consumer, tdm_smem_a, dot_a_layout, tdm_smem_b, dot_b_layout, zeros, 0, NUM_BUFFERS, False)
            acc += zeros * cur_a_scale[:, None] * cur_b_scale[None, :]
            zeros = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)


        c = acc.to(c_ptr.type.element_ty)

        # # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
            0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, wmma_layout)
        )
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
            0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, wmma_layout)
        )
        c_offs = (
            stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        gl.amd.cdna4.buffer_store(
            stored_value=c, ptr=c_ptr, offsets=c_offs, mask=c_mask
        )
        # gl.store(
        #     c_ptr + c_offs,
        #     c,
        #     mask=c_mask,
        # )


@gluon.jit
def _gemm_a8w8_blockscale_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    M,
    N,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_m,
    stride_c_out_n,
    BLOCK_SIZE_M: gl.constexpr,  # Note: Can be distinct from GEMM block size
    BLOCK_SIZE_N: gl.constexpr,
    ACTUAL_KSPLIT: gl.constexpr,
    MAX_KSPLIT: gl.constexpr,
):

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)

    blocked_read: gl.constexpr = gl.BlockedLayout(  # (MAX_KSPLIT, BLOCK_M, BLOCK_N)
        size_per_thread=[1, 1, 4],
        threads_per_warp=[1, 8, 4],
        warps_per_cta=[1, 4, 1],
        order=[2, 1, 0],
    )

    # blocked_write: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread=[1, 4], # (BLOCK_M, BLOCK_N)
    #     threads_per_warp=[4, 8],
    #     warps_per_cta=[4, 1],
    #     order=[1, 0],
    # )

    offs_m = pid_m * BLOCK_SIZE_M + gl.arange(
        0,
        BLOCK_SIZE_M,  # keep dim 1
        gl.SliceLayout(0, gl.SliceLayout(2, blocked_read)),
    )
    offs_n = pid_n * BLOCK_SIZE_N + gl.arange(
        0,
        BLOCK_SIZE_N,  # keep dim 2
        gl.SliceLayout(0, gl.SliceLayout(1, blocked_read)),
    )
    offs_k = gl.arange(
        0, MAX_KSPLIT, gl.SliceLayout(1, gl.SliceLayout(2, blocked_read))  # keep dim 0
    )
    c_in_offs = (
        (offs_k[:, None, None] * stride_c_in_k)
        + (offs_m[None, :, None] * stride_c_in_m)
        + (offs_n[None, None, :] * stride_c_in_n)
    )
    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c_in_mask = (offs_m[None, :, None] < M) & (offs_n[None, None, :] < N)
        c = gl.amd.cdna4.buffer_load(c_in_ptr, c_in_offs, mask=c_in_mask, cache=".ca")
    else:
        c_in_mask = (
            (offs_m[None, :, None] < M)
            & (offs_n[None, None, :] < N)
            & (offs_k[:, None, None] < ACTUAL_KSPLIT)
        )
        c = gl.amd.cdna4.buffer_load(
            c_in_ptr, c_in_offs, mask=c_in_mask, cache=".ca"
        )  # , other=0.0)
    c = tl.sum(c, 0)

    c = c.to(c_out_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, gl.SliceLayout(1, gl.SliceLayout(0, blocked_read))
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, gl.SliceLayout(0, gl.SliceLayout(0, blocked_read))
    )
    c_out_offs = (offs_cm[:, None] * stride_c_out_m) + (
        offs_cn[None, :] * stride_c_out_n
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    gl.amd.cdna4.buffer_store(
        stored_value=c, ptr=c_out_ptr, offsets=c_out_offs, mask=c_mask
    )


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_arch()
        if int(dev.split("gfx")[1]) < 950:
            raise ValueError(
                "Gluon implementation is not supported on this device (requires CDNA4)."
            )
        _get_config._config_dict = {}
        fpath = (
            f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/gfx1250-GEMM-A8W8_BLOCKSCALE.json"
        )
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_arch()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/{dev}-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config

    # Config keys should be named M_LEQ_<bound> or "any"
    bounds = []
    for setting in _get_config._config_dict[key].keys():
        potential_block_m = setting.replace("M_LEQ_", "")
        if potential_block_m.isnumeric():
            bounds.append(int(potential_block_m))

    for bound in bounds:
        if M <= bound and f"M_LEQ_{bound}" in _get_config._config_dict[key]:
            config = _get_config._config_dict[key][f"M_LEQ_{bound}"]
            break
        else:
            config = _get_config._config_dict[key]["any"]

    config = (
        config.copy()
    )  # avoid later inplace modification from interacting with cached config

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)

    return config


def gemm_a8w8_blockscale(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes the 8 bit matmul Y = X x WT using the block-scale quantization approach.

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - X_scale: Scale tensor for X with shape (M, *scale_k).
    - W_scale: Scale tensor for W with shape (**scale_n, *scale_k).

    Returns:
    - Y: The output matrix with shape (M, N).

    *scale_k = (K + scale_block_size_k - 1) // scale_block_size_k
    **scale_n = (N + scale_block_size_n - 1) // scale_block_size_n
    """
    _LOGGER.info(
        f"GEMM_A8W8_BLOCKSCALE: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    M, K = x.shape
    N, K = w.shape

    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"

    # Transpose w and w_scale
    w = w.T
    w_scale = w_scale.T

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    # Scale block sizes
    # TODO: need a better way to pass scale block sizes around
    config["GROUP_K"] = triton.next_power_of_2(triton.cdiv(K, w_scale.shape[0]))
    config["GROUP_N"] = triton.next_power_of_2(triton.cdiv(N, w_scale.shape[1]))

    if config["NUM_KSPLIT"] == 1:
        assert (
            config["GROUP_K"] == config["BLOCK_SIZE_K"]
        ), f"GROUP_K: {config['GROUP_K']} must equal BLOCK_SIZE_K: {config['BLOCK_SIZE_K']} when not using KSPLIT"

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=y.device
        )
    else:
        y_pp = None

    # grid = (config["NUM_KSPLIT"], triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    NUM_WARPS=config["num_warps"]
    warp_bases = [(0, 1)]
    for i in range(int(math.log2(NUM_WARPS // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)
    #print(x)
    # print(w)
    # print(x_scale)
    # print(w_scale)
    # print(y)
    _gemm_a8w8_blockscale_kernel[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scale,
        w_scale,
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
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        NUM_WARPS=config["num_warps"],
        warp_bases=warp_bases,
        **config,
    )
    print("complete")
    print(y_pp)

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )

        _gemm_a8w8_blockscale_reduce_kernel[grid_reduce](
            y_pp,
            y,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
        )

    return y
