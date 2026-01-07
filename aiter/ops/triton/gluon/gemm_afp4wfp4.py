from typing import Optional
import functools
import json
import triton
import torch
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_LOGGER = AiterTritonLogger()

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
def _gemm_afp4wfp4_kernel(
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
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    EVEN_K: gl.constexpr,
    num_warps: gl.constexpr,
    num_stages: gl.constexpr,
    waves_per_eu: gl.constexpr,
    matrix_instr_nonkdim: gl.constexpr,
    cache_modifier: gl.constexpr,
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
    # remap so that XCDs get continous chunks of pids (of CHUNK_SIZE).
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
    SCALE_GROUP_SIZE: gl.constexpr = 32

    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )

    blocked_scales: gl.constexpr = gl.BlockedLayout(
        size_per_thread= [4, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )

    linear_as: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 2], [0, 4], [64, 0], [128, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]],
        warp_bases=[[0, 0], [0, 0], [32, 0]],
        block_bases=[],
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_K//SCALE_GROUP_SIZE],        
    )

    linear_bs: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 2], [0, 4], [128, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]],
        warp_bases=[[32, 0], [64, 0], [0, 0]],
        block_bases=[],
        shape=[BLOCK_SIZE_N, BLOCK_SIZE_K//SCALE_GROUP_SIZE],        
    )

    linear_mn: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 16], [0, 128], [64, 0], [128, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]],
        warp_bases=[[0, 32], [0, 64], [32, 0]],
        block_bases=[],
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],    
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )

    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0, 1]
    )

    shared_scales: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0, 1]
    )
    
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32],
        transposed=True,
        warps_per_cta=[2, num_warps//2],
    )

    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:

        num_k_iter = gl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

        # Create pointers for first block of A and B input matrices
        # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
        offs_ak = gl.arange(0, BLOCK_SIZE_K // 2, layout=gl.SliceLayout(0, blocked_mk))
        offs_ak_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_ak
        offs_bk = gl.arange(0, BLOCK_SIZE_K // 2, layout=gl.SliceLayout(1, blocked_kn))
        offs_bk_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_bk
        offs_am = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk))) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn))) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_ak_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_bk_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )
        
        # Create pointers for the first block of A and B scales
        offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + gl.arange(
            0, BLOCK_SIZE_K // SCALE_GROUP_SIZE, layout=gl.SliceLayout(0, blocked_scales)
        )

        offs_as = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_scales))) % M
        offs_bs = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(1, blocked_scales))) % N

        
        a_scale_ptrs = (
            a_scales_ptr + offs_as[:, None] * stride_asm + offs_ks[None, :] * stride_ask
        )
        # B scales are N x K even though B operand is K x N.
        b_scale_ptrs = (
            b_scales_ptr + offs_bs[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk
        )

        # Create shared memories
        smem_a = gl.allocate_shared_memory(
            a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K//2], layout=shared_a
        )
        smem_b = gl.allocate_shared_memory(
            b_ptr.type.element_ty, [BLOCK_SIZE_K//2, BLOCK_SIZE_N], layout=shared_b
        )

        smem_as = gl.allocate_shared_memory(
            a_scales_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE], layout=shared_scales
        )
        smem_bs = gl.allocate_shared_memory(
            b_scales_ptr.type.element_ty, [BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE], layout=shared_scales
        )        

        a_scales = gl.load(a_scale_ptrs)
        b_scales = gl.load(b_scale_ptrs, cache_modifier=cache_modifier)

        if EVEN_K:
                a = gl.load(a_ptrs)
                b = gl.load(b_ptrs, cache_modifier=cache_modifier)
        else:
            a = gl.load(
                a_ptrs, mask=offs_ak[None, :] < K - pid_k * SPLITK_BLOCK_SIZE // 2, other=0
            )
            b = gl.load(
                b_ptrs,
                mask=offs_bk[:, None] < K - pid_k * SPLITK_BLOCK_SIZE // 2,
                other=0,
                cache_modifier=cache_modifier,
            )

        smem_as.store(a_scales)
        smem_bs.store(b_scales)
        smem_a.store(a)
        smem_b.store(b)

        accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter - 1):
            
            #advance pointers
            a_ptrs += (BLOCK_SIZE_K // 2) * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
            a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
            b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk
            
            a_scales = gl.load(a_scale_ptrs)
            curr_a_scales = smem_as.load(layout=linear_as)
            
            b_scales = gl.load(b_scale_ptrs, cache_modifier=cache_modifier)
            curr_b_scales = smem_bs.load(layout=linear_bs)

            if EVEN_K:
                a = gl.load(a_ptrs)
            else:
                a = gl.load(
                    a_ptrs, mask=offs_ak[None, :] < K - (k+1) * (BLOCK_SIZE_K // 2), other=0
                )
            curr_a = smem_a.load(layout=dot_a_layout)

            if EVEN_K:
                b = gl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                b = gl.load(
                    b_ptrs,
                    mask=offs_bk[:, None] < K - (k+1) * (BLOCK_SIZE_K // 2),
                    other=0,
                    cache_modifier=cache_modifier,
                )
            curr_b = smem_b.load(layout=dot_b_layout)

            accumulator = gl.amd.cdna4.mfma_scaled(
                a=curr_a,
                a_scale=curr_a_scales,
                a_format='e2m1',
                b=curr_b,
                b_scale=curr_b_scales,
                b_format='e2m1',
                acc=accumulator,
            )

            smem_as.store(a_scales)
            smem_bs.store(b_scales)
            smem_a.store(a)
            smem_b.store(b)

        curr_a_scales = smem_as.load(layout=linear_as)
        curr_b_scales = smem_bs.load(layout=linear_bs)
        curr_a = smem_a.load(layout=dot_a_layout)
        curr_b = smem_b.load(layout=dot_b_layout)

        accumulator = gl.amd.cdna4.mfma_scaled(
            a=curr_a,
            a_scale=curr_a_scales,
            a_format='e2m1',
            b=curr_b,
            b_scale=curr_b_scales,
            b_format='e2m1',
            acc=accumulator,
        )

        c = accumulator.to(c_ptr.type.element_ty)
        c = gl.convert_layout(c, layout=linear_mn, assert_trivial=False)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, linear_mn)).to(gl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, linear_mn)).to(gl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        gl.store(c_ptrs, c, mask=c_mask)

@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):

    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_arch()
        if dev != "gfx950":
            raise ValueError(
                "Gluon implementation is not supported on this device (requires CDNA4)."
            )
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/{dev}-GEMM-AFP4WFP4.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict = config

    return _get_config._config_dict["any"]

def gemm_afp4wfp4(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[str] = None,
    skip_reduce: Optional[bool] = False,
) -> torch.Tensor:
    """
    Computes matrix multiplication Y = X @ W^T with FP4 activations and FP4 weights.

    Args:
        x (torch.Tensor): FP4 E2M1 input matrix with shape (M, K//2).
        w (torch.Tensor): FP4 E2M1 weight matrix with shape (N, K//2), internally transposed.
        x_scales (torch.Tensor): E8M0 per-group scale for x with shape (M, K//32).
            One scale per 32 elements in K dimension.
        w_scales (torch.Tensor): E8M0 per-group scale for w with shape (N, K//32).
            One scale per 32 elements in K dimension.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).
        skip_reduce (Optional[bool]): skip reduction, y becomes (SPK, M, N) where SPK is determined by config

    Returns:
        y (torch.Tensor): Output with shape (M, N) or (SPK, M, N).
    """
    _LOGGER.info(
        f"GEMM_AFPWFP4: x.shape={tuple(x.shape)} w.shape={tuple(w.shape)} x_scale={tuple(x_scales.shape)} w_scale={tuple(w_scales.shape)} "
    )

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

    M, K = x.shape
    N, K = w.shape

    # Transpose w
    w = w.T

    if config is None:
        config = _get_config(M, N, K)

    # if config["NUM_KSPLIT"] > 1:
    #     SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
    #         K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
    #     )

    #     config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
    #     config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
    #     config["NUM_KSPLIT"] = NUM_KSPLIT

    # if config["BLOCK_SIZE_K"] >= 2 * K:
    #     config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
    #     config["SPLITK_BLOCK_SIZE"] = 2 * K
    #     config["NUM_KSPLIT"] = 1
    # config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 128)

    # if config["NUM_KSPLIT"] > 1:
    #     if _USE_GEMM_SPLITK_BF16:
    #         y_pp = torch.empty(
    #             (config["NUM_KSPLIT"], M, N), dtype=y.dtype, device=x.device
    #         )
    #     else:
    #         y_pp = torch.empty(
    #             (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=x.device
    #         )
    # else:
    #     config["SPLITK_BLOCK_SIZE"] = 2 * K
    #     y_pp = None
    config["SPLITK_BLOCK_SIZE"] = 2 * K
    y_pp = None

    if y is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )

    _gemm_afp4wfp4_kernel[grid](
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

    # if config["NUM_KSPLIT"] > 1:
    #     if skip_reduce:
    #         return y_pp

    #     REDUCE_BLOCK_SIZE_M = 16
    #     # TODO: Need to debug - REDUCE_BLOCK_SIZE_N=128 with fp32 partials fails
    #     # NOTE: REDUCE_BLOCK_SIZE_N=16 gives best perf with fp32 partials and
    #     # REDUCE_BLOCK_SIZE_N=128 gives best perf with bf16 partials
    #     REDUCE_BLOCK_SIZE_N = 128 if _USE_GEMM_SPLITK_BF16 else 64
    #     ACTUAL_KSPLIT = triton.cdiv(K, (config["SPLITK_BLOCK_SIZE"] // 2))

    #     grid_reduce = (
    #         triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
    #         triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
    #     )
    #     _gemm_afp4wfp4_reduce_kernel[grid_reduce](
    #         y_pp,
    #         y,
    #         M,
    #         N,
    #         y_pp.stride(0),
    #         y_pp.stride(1),
    #         y_pp.stride(2),
    #         y.stride(0),
    #         y.stride(1),
    #         REDUCE_BLOCK_SIZE_M,
    #         REDUCE_BLOCK_SIZE_N,
    #         ACTUAL_KSPLIT,
    #         triton.next_power_of_2(config["NUM_KSPLIT"]),
    #     )

    return y