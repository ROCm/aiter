# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Grouped contiguous-M A8W4 preshuffle MoE GEMM for gfx1250 (TDM pipeline)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Constexpr

from .gemm1_consumer_gfx1250 import emit_gemm_a8w4_tile, gemm_a8w4_tile_config
from .kernels_common import ceildiv
from .tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE,
)

TDM_DESCRIPTOR_VERSION = 1


@flyc.jit
def launch_gemm_a8w4_tdm(
    arg_c: fx.Tensor,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Tensor,
    arg_scale_b: fx.Tensor,
    i32_m: fx.Int32,
    stream: fx.Stream,
    N: fx.Int32,
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    out_is_f16: Constexpr[int],
    num_buffers: Constexpr[int],
    a_is_fp4: Constexpr[int],
    arg_m_tile_map: fx.Pointer,
    n_experts: Constexpr[int],
    stage1_act: Constexpr[int],
    has_bias: Constexpr[int],
    arg_bias: fx.Pointer,
    f32_swiglu_limit: fx.Float32,
    stage1_quant_out: Constexpr[int] = 0,
    quant_wmma_rep: Constexpr[int] = 1,
    arg_quant_scale: fx.Tensor = None,
    cluster_n: Constexpr[int] = 1,
    next_stage_prefetch: Constexpr[int] = 0,
    num_waves_per_tensor_tdm: Constexpr[int] = 2,
    tdm_as_in_prologue: Constexpr[int] = 0,
    tdm_b_th: Constexpr[int] = 0,
    enable_ep_scatter: Constexpr[int] = 0,
    ep_arena_handle: Constexpr[int] = 0,
    ep_combine_input_offset: Constexpr[int] = 0,
    ep_slot_stride_bytes: Constexpr[int] = 0,
    ep_destination_stride: Constexpr[int] = 0,
    ep_world_size: Constexpr[int] = 0,
    ep_quant_bits: Constexpr[int] = 0,
    arg_ep_row_map: fx.Tensor = None,
    f32_situ_beta: fx.Float32 = 1.0,
    f32_situ_linear_beta: fx.Float32 = 1.0,
    row_major_ascale: Constexpr[int] = 0,
    a_row_stride_bytes: Constexpr[int] = 0,
    a_scale_row_stride_bytes: Constexpr[int] = 0,
):
    """Launch the grouped contiguous-M a8w4 MoE GEMM for gfx1250.

    ``cluster_n`` > 1 launches (cluster_n, 1, 1) workgroup clusters whose peers
    all share one m_tile (and therefore one expert) and differ only in n_tile, so
    one A / A-scale load can serve the whole cluster.

    No cluster barrier is emitted, and none is needed: a non-zero workgroup_mask
    turns the load into CLUSTER_LOAD_ASYNC, which rendezvouses with the peers the
    mask names, and each workgroup's own s_wait_tensorcnt still covers its own
    LDS. That is the same protocol as opus (see csrc/opus_gemm/include/gfx1250/
    opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh), which
    emits s_barrier -3 only for a 2D cluster whose mask is a strided group; for a
    1-D cluster like this one the mask is contiguous, the barrier is unnecessary,
    and on a thin 1-D cluster it can hang on co-residency.

    The rendezvous replaces drift bounding with two hard preconditions, and
    breaking either hangs rather than corrupts:

    1. Every peer issues the same number of pairwise-matching multicast loads.
       This holds because K_TILES is a compile-time constant and the
       ``expert < n_experts`` skip is cluster-uniform: peers share m_tile, hence
       expert, so they all skip or none do.
    2. The grid fills every cluster exactly, i.e. ceil(N/tile_n) % cluster_n == 0.
       That cannot be checked here -- inside @flyc.jit ``N`` is a traced value, so
       a Python ``if`` on it becomes a traced branch rather than a host-side
       check -- so the callers that choose cluster_n enforce it
       (batched_gemm_mxfp4._pick_cluster_n and its assert).
    """
    # Every compile-time constant of the tile body, plus the symbol name and
    # workgroup size this launch needs, come from the shared config so the
    # mega-kernel scheduler derives them the same way.
    tile_cfg = gemm_a8w4_tile_config(
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        out_is_f16=out_is_f16,
        num_buffers=num_buffers,
        a_is_fp4=a_is_fp4,
        n_experts=n_experts,
        stage1_act=stage1_act,
        has_bias=has_bias,
        stage1_quant_out=stage1_quant_out,
        quant_wmma_rep=quant_wmma_rep,
        cluster_n=cluster_n,
        next_stage_prefetch=next_stage_prefetch,
        num_waves_per_tensor_tdm=num_waves_per_tensor_tdm,
        tdm_as_in_prologue=tdm_as_in_prologue,
        tdm_b_th=tdm_b_th,
        enable_ep_scatter=enable_ep_scatter,
        ep_arena_handle=ep_arena_handle,
        ep_combine_input_offset=ep_combine_input_offset,
        ep_slot_stride_bytes=ep_slot_stride_bytes,
        ep_destination_stride=ep_destination_stride,
        ep_world_size=ep_world_size,
        ep_quant_bits=ep_quant_bits,
        row_major_ascale=row_major_ascale,
        a_row_stride_bytes=a_row_stride_bytes,
        a_scale_row_stride_bytes=a_scale_row_stride_bytes,
    )
    block = tile_cfg.block
    next_stage_on = tile_cfg.next_stage_on
    cache_tag = (
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        out_is_f16,
        num_buffers,
        a_is_fp4,
        n_experts,
        stage1_act,
        has_bias,
        TDM_DESCRIPTOR_VERSION,
        stage1_quant_out,
        quant_wmma_rep,
        cluster_n,
        next_stage_on,
        num_waves_per_tensor_tdm,
        tdm_as_in_prologue,
        tdm_b_th,
        enable_ep_scatter,
        ep_arena_handle,
        ep_combine_input_offset,
        ep_slot_stride_bytes,
        ep_destination_stride,
        ep_world_size,
        row_major_ascale,
        a_row_stride_bytes,
        a_scale_row_stride_bytes,
        ep_quant_bits,
    )
    _ = cache_tag
    _kname = tile_cfg.kernel_name

    @flyc.kernel(name=_kname, known_block_size=[block, 1, 1])
    def kernel(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        arg_m_tile_map: fx.Pointer,
        arg_bias: fx.Pointer,
        arg_quant_scale: fx.Pointer,
        arg_ep_row_map: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        f32_swiglu_limit: fx.Float32,
        f32_situ_beta: fx.Float32,
        f32_situ_linear_beta: fx.Float32,
    ):
        emit_gemm_a8w4_tile(
            arg_c=arg_c,
            arg_a=arg_a,
            arg_b=arg_b,
            arg_scale_a=arg_scale_a,
            arg_scale_b=arg_scale_b,
            arg_m_tile_map=arg_m_tile_map,
            arg_bias=arg_bias,
            arg_quant_scale=arg_quant_scale,
            arg_ep_row_map=arg_ep_row_map,
            i32_m=i32_m,
            i32_n=i32_n,
            f32_swiglu_limit=f32_swiglu_limit,
            f32_situ_beta=f32_situ_beta,
            f32_situ_linear_beta=f32_situ_linear_beta,
            bid_x=fx.block_idx.x,
            **tile_cfg.consts,
        )

    m_tiles = ceildiv(i32_m, tile_m)
    n_tiles = ceildiv(N, tile_n)
    if arg_ep_row_map is None:
        arg_ep_row_map = arg_c
    if arg_quant_scale is None:
        arg_quant_scale = arg_c
    kargs = (
        fx.get_iter(arg_c),
        arg_a,
        arg_b,
        fx.get_iter(arg_scale_a),
        fx.get_iter(arg_scale_b),
        arg_m_tile_map,
        arg_bias,
        fx.get_iter(arg_quant_scale),
        fx.get_iter(arg_ep_row_map),
        i32_m,
        N,
        f32_swiglu_limit,
        f32_situ_beta,
        f32_situ_linear_beta,
    )
    grid = (m_tiles * n_tiles, 1, 1)
    if cluster_n > 1:
        # Geometry must reach BOTH the definition and the launch site, or the
        # cluster never forms and the TDM loads silently fall back to per-load.
        kernel(
            *kargs,
            value_attrs={"rocdl.cluster_dims": f"{cluster_n},1,1"},
        ).launch(
            grid=grid,
            block=(block, 1, 1),
            stream=stream,
            cluster=(cluster_n, 1, 1),
        )
    else:
        kernel(*kargs).launch(grid=grid, block=(block, 1, 1), stream=stream)


launch_gemm_a8w4_tdm.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE,
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
