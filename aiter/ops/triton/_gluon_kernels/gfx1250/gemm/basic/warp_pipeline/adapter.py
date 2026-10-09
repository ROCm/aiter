# SPDX-License-Identifier: MIT
# Layout/launch adapter derived from the pinned Triton CDNA5 example.
"""Opt-in upstream warp-pipeline GEMM. Never changes AITER default dispatch.

Inputs are contiguous E4M3 A[M,K], W[N,K], and row-major uint8 E8M0
scales A[M,K/128], W[N/128,K/128]. Prepare once outside graph timing.
Native K128 A scales are compactly preshuffled without duplication; W scales are used directly.
Only M-contiguous K128 scales and padded two-buffer output stores are supported.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import triton
import triton.experimental.gluon.language as gl
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental.gluon.language.amd.cdna5 import PartitionedSharedLayout


def _example():
    from .. import gemm_warp_pipeline_cdna5

    return gemm_warp_pipeline_cdna5


@dataclass
class PreparedWarpPipeline:
    launch: object
    output: torch.Tensor
    inputs: tuple
    tile: str
    cluster: str

    def __call__(self):
        self.launch()
        return self.output


def prepare_gemm_warp_pipeline(
    a,
    w,
    a_scales,
    w_scales,
    *,
    tile="256x256x256",
    cluster="2x2",
    flat_pipeline=False,
    out=None,
    a_scales_transposed=False,
):
    """Prepare the upstream kernel for canonical, unshuffled block-128 inputs."""
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[1]:
        raise ValueError("Expected A[M,K] and W[N,K]")
    m, k = a.shape
    n = w.shape[0]
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx1250":
        raise ValueError(f"Requires gfx1250; got {target}")
    if any(
        t.device != a.device or not t.is_contiguous()
        for t in (a, w, a_scales, w_scales)
    ):
        raise ValueError("Inputs must be contiguous on the same GPU")
    if a.dtype != torch.float8_e4m3fn or w.dtype != a.dtype:
        raise ValueError("A and W must be float8_e4m3fn")
    if a_scales.dtype != torch.uint8 or w_scales.dtype != torch.uint8:
        raise ValueError("Scales must contain uint8 E8M0 codes")
    if tuple(a_scales.shape) != (m, k // 128) or tuple(w_scales.shape) != (
        n // 128,
        k // 128,
    ):
        raise ValueError("Expected block-128 row-major scales")
    example = _example()
    tile_name, cluster_name = tile, cluster
    tile, block_k, cluster = example.validate_gemm_shape(
        "mxfp8", m, n, k, tile, cluster
    )
    args = SimpleNamespace(M=m, N=n, K=k, mxfp8_output_buffers=2)
    bk256_kernel = (
        example.mxfp8_gemm_warp_pipeline_flat
        if flat_pipeline
        else example.mxfp8_gemm_warp_pipeline
    )
    tiled_kernel = (
        example.mxfp8_gemm_warp_pipeline_bk512_flat
        if flat_pipeline
        else example.mxfp8_gemm_warp_pipeline_bk512
    )
    if block_k == 1024:
        tiled_kernel = (
            example.mxfp8_gemm_warp_pipeline_bk1024_flat
            if flat_pipeline
            else example.mxfp8_gemm_warp_pipeline_bk1024
        )
    block_m = block_n = tile * cluster
    a_d, b_d = a, w
    as_d = (
        a_scales.view(k // 128, m) if a_scales_transposed else a_scales.T.contiguous()
    )
    bs_d = w_scales
    c_d = (
        torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        if out is None
        else out
    )
    if (
        c_d.shape != (m, n)
        or c_d.dtype != torch.bfloat16
        or c_d.device != a.device
        or not c_d.is_contiguous()
    ):
        raise ValueError("Output must be contiguous BF16 [M,N] on the same GPU")
    # First derive the per-CTA WMMA ownership, then attach the cluster CGA bases.

    cga = make_cga_layout([cluster, cluster], [cluster, cluster], [0, 1])
    output_cga = tuple(tuple(basis) + (0, 0) for basis in cga)
    slice_n = tile // 2 if block_k == 256 else tile
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 16]], [block_m, block_k], [1, 0]
    )
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 16]], [block_n, block_k], [1, 0]
    )
    _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(
        block_m,
        block_n,
        local_a,
        local_b,
        8,
        [16, 16, 128],
        a_transposed=False,
        b_transposed=True,
        slice_m=tile,
        slice_n=slice_n,
    )
    load_wmma = gl.amd.AMDWMMALayout(
        3,
        local_wmma.transposed,
        local_wmma.warp_bases,
        local_wmma.reg_bases,
        local_wmma.instr_shape,
        cga,
    )
    dot_a = gl.DotOperandLayout(0, load_wmma, 16)
    dot_b = gl.DotOperandLayout(1, load_wmma, 16)
    cga_a = dot_a.cga_layout
    cga_b = tuple((basis[1], basis[0]) for basis in dot_b.cga_layout)
    padded_a = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 16]], [block_m, block_k], [1, 0], cga_a
    )
    padded_b = gl.PaddedSharedLayout.with_identity_for(
        [[block_k, 16]], [block_n, block_k], [1, 0], cga_b
    )
    shared_a, shared_b, _ = gl.amd.cdna5.make_partitioned_dot_layouts(
        tile,
        tile,
        padded_a,
        padded_b,
        8,
        [16, 16, 128],
        a_transposed=False,
        b_transposed=True,
        slice_m=tile,
        slice_n=slice_n,
    )
    if block_k == 256:
        # BK256 splits N into two accumulators; legacy mode uses packed scales.
        packed_n = block_n // 2
        local_b = gl.PaddedSharedLayout.with_identity_for(
            [[256, 16]], [packed_n, 256], [1, 0]
        )
        _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(
            block_m,
            packed_n,
            local_a,
            local_b,
            8,
            [16, 16, 128],
            a_transposed=False,
            b_transposed=True,
            slice_m=tile,
            slice_n=slice_n,
        )
        wmma = gl.amd.AMDWMMALayout(
            3,
            local_wmma.transposed,
            local_wmma.warp_bases,
            local_wmma.reg_bases,
            local_wmma.instr_shape,
            cga,
        )
        shared_as = gl.PaddedSharedLayout.with_identity_for(
            [[256, 8]], [block_m // 128, 1024], [1, 0], cga_a
        )
        shared_bs = gl.PaddedSharedLayout.with_identity_for(
            [[256, 8]], [block_n // 128, 1024], [1, 0], cga_b
        )
        # One parent B allocation contains both N halves for every CTA column.
        inner = shared_b.partition_layout
        parent_cga = tuple((basis[0], 0, basis[1]) for basis in inner.cga_layout)
        padded = gl.PaddedSharedLayout.with_identity_for(
            [[256, 16]], [cluster, 64, 256], [2, 1, 0], parent_cga
        )
        b2_shared = PartitionedSharedLayout(2, 2, 1, padded)
        b2_load = gl.DistributedLinearLayout(
            reg_bases=[
                [1, 0, 0],
                [2, 0, 0],
                [4, 0, 0],
                [8, 0, 0],
                [32, 0, 0],
                [64, 0, 0],
                [0, 0, 16],
                [0, 0, 32],
            ],
            lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [16, 0, 0]],
            warp_bases=[[0, 0, 64], [0, 0, 0], [0, 0, 0]],
            block_bases=[[0, 0, 0]] * (cluster.bit_length() - 1)
            + [[0, 1 << i, 0] for i in range(cluster.bit_length() - 1)],
            shape=[128, cluster, 128],
        )
    else:
        # BK512/BK1024 use a full accumulator; legacy scales are unshuffled.
        wmma = load_wmma
        shared_as = gl.PaddedSharedLayout.with_identity_for(
            [[16, 8]], [block_m, block_k // 32], [1, 0], cga_a
        )
        shared_bs = gl.PaddedSharedLayout.with_identity_for(
            [[16, 8]], [block_n, block_k // 32], [1, 0], cga_b
        )
    mcontiguous_b_cga = tuple(
        (basis[0] // max(1, 128 // tile), basis[1]) for basis in cga_b
    )
    shared_as = gl.PaddedSharedLayout.with_identity_for(
        [[64, 8]], [block_m, block_k // 128], [1, 0], cga_a
    )
    shared_bs = gl.PaddedSharedLayout.with_identity_for(
        [[64, 8]], [block_n // 128, block_k // 128], [1, 0], mcontiguous_b_cga
    )
    grid = (triton.cdiv(args.M, block_m) * triton.cdiv(args.N, block_n), 1)

    def launch(a_d=a_d, b_d=b_d, c_d=c_d, as_d=as_d, bs_d=bs_d):
        if block_k in (512, 1024):
            return tiled_kernel[grid](
                a_d,
                b_d,
                c_d,
                as_d,
                bs_d,
                args.M,
                args.N,
                args.K,
                a_d.stride(0),
                a_d.stride(1),
                b_d.stride(1),
                b_d.stride(0),
                c_d.stride(0),
                c_d.stride(1),
                as_d.stride(0),
                bs_d.stride(0),
                GRID_MN=grid[0],
                SHARED_A=shared_a,
                SHARED_B=shared_b,
                SHARED_AS=shared_as,
                SHARED_BS=shared_bs,
                WMMA=wmma,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                CLUSTER=cluster,
                num_warps=8,
                num_ctas=cluster * cluster,
                waves_per_eu=2,
            )
        return bk256_kernel[grid](
            a_d,
            b_d,
            c_d,
            as_d,
            bs_d,
            args.M,
            args.N,
            args.K,
            a_d.stride(0),
            a_d.stride(1),
            b_d.stride(1),
            b_d.stride(0),
            c_d.stride(0),
            c_d.stride(1),
            as_d.stride(0),
            bs_d.stride(0),
            GRID_MN=grid[0],
            SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B2=b2_shared,
            SHARED_SCALE_A=shared_as,
            SHARED_SCALE_B=shared_bs,
            WMMA_LAYOUT=wmma,
            LOAD_WMMA_LAYOUT=load_wmma,
            B2_LOAD_LAYOUT=b2_load,
            OUTPUT_CGA_LAYOUT=output_cga,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            CTA_M=cluster,
            CTA_N=cluster,
            num_warps=8,
            waves_per_eu=2,
            num_ctas=cluster * cluster,
        )

    return PreparedWarpPipeline(
        launch, c_d, (a_d, b_d, c_d, as_d, bs_d), tile_name, cluster_name
    )


def gemm_warp_pipeline_cdna5(a, w, a_scales, w_scales, **kwargs):
    """Convenience API including scale preparation; use prepare_* for timing."""
    return prepare_gemm_warp_pipeline(a, w, a_scales, w_scales, **kwargs)()
