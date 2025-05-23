from typing import Optional
import os
import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils.pid_preprocessing import pid_grid, remap_xcd

DEBUG = False

@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _gemm_a8wfp4_kernel(
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    RAW_MASKED_LOADS: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A is in fp8 e4m3 format.
    B is in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        remap_xcd(pid, GRID_MN)

        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)
    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32

    if (pid_k * SPLITK_BLOCK_SIZE) < K:
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        # Set up base A offsets
        offs_am_raw = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_am = offs_am_raw % M
        
        # Load A scales once (they're per-row)
        a_scale_ptrs = a_scales_ptr + offs_am * stride_asm
        if RAW_MASKED_LOADS:
            a_scale_mask = (offs_am < M)
            a_scales = tl.load(a_scale_ptrs, mask=a_scale_mask)
        else:
            a_scales = tl.load(a_scale_ptrs)
        a_ones_scale = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_K//SCALE_GROUP_SIZE), 127, dtype=tl.uint8) # 1.0 in e8m0

        # Set up base B offsets  
        offs_bn_raw = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        offs_bn = offs_bn_raw % N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, num_k_iter):
            # Load A inside the loop with correct K offset
            offs_ak = tl.arange(0, BLOCK_SIZE_K) + k * BLOCK_SIZE_K  # Add k offset
            offs_ak_split = pid_k * SPLITK_BLOCK_SIZE + offs_ak
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_ak_split[None, :] * stride_ak
            )
            if RAW_MASKED_LOADS:
                a_mask = (offs_am_raw[:, None] < M) & (offs_ak_split[None, :] < K)
                a = tl.load(a_ptrs, mask=a_mask)
            else:
                a = tl.load(a_ptrs)
            
            # B loading stays mostly the same, but fix the offsets
            offs_bk = tl.arange(0, BLOCK_SIZE_K // 2) + k * (BLOCK_SIZE_K // 2)  # Add k offset
            offs_bk_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_bk
            b_ptrs = b_ptr + (
                offs_bk_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )
            offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE) + tl.arange(
                0, BLOCK_SIZE_K // SCALE_GROUP_SIZE
            )
            b_scale_ptrs = (
                b_scales_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk
            )
            if RAW_MASKED_LOADS:
                b_k_mask = (offs_bk_split[:, None] < (K // 2))
                b_n_mask = offs_bn_raw[None, :] < N
                b_mask = b_k_mask & b_n_mask
                if EVEN_K:
                    b = tl.load(b_ptrs, mask=b_mask, cache_modifier=cache_modifier)
                else:
                    b = tl.load(b_ptrs, mask=b_mask, other=0)
                bs_k_mask = (offs_ks[None, :] < (K // SCALE_GROUP_SIZE))
                bs_n_scale_mask = (offs_bn_raw[:, None] < N)
                bs_mask = bs_k_mask & bs_n_scale_mask
                b_scales = tl.load(b_scale_ptrs, mask=bs_mask, other=0)
            else:
                if EVEN_K:
                    b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                else:
                    b_mask = offs_bk[:, None] < K - k * (BLOCK_SIZE_K // 2)
                    b = tl.load(b_ptrs, mask=b_mask, other=0)
                b_scales = tl.load(b_scale_ptrs)
            accumulator += tl.dot_scaled(a, a_ones_scale, "e4m3", b, b_scales, "e2m1")

        # Scale by a_scales at the end
        c = (accumulator * a_scales[:, None]).to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def _gemm_afp4_wfp4_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    M,
    N,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_m,
    stride_c_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
):

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, MAX_KSPLIT)
    c_in_ptrs = (
        c_in_ptr
        + (offs_k[:, None, None] * stride_c_in_k)
        + (offs_m[None, :, None] * stride_c_in_m)
        + (offs_n[None, None, :] * stride_c_in_n)
    )

    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c = tl.load(c_in_ptrs)
    else:
        c = tl.load(c_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT)
    c = tl.sum(c, axis=0)

    c = c.to(c_out_ptr.type.element_ty)

    c_out_ptrs = (
        c_out_ptr
        + (offs_m[:, None] * stride_c_out_m)
        + (offs_n[None, :] * stride_c_out_n)
    )

    tl.store(c_out_ptrs, c)


def get_splitk(K: int, BLOCK_SIZE_K: int, NUM_KSPLIT: int):
    # heuristics for make "EVEN_K == True" as much as possible
    NUM_KSPLIT_STEP = 4
    BLOCK_SIZE_K_STEP = 4
    SPLITK_BLOCK_SIZE = (
        triton.cdiv((2 * triton.cdiv(K, NUM_KSPLIT)), BLOCK_SIZE_K) * BLOCK_SIZE_K
    )
    while NUM_KSPLIT > 1 and BLOCK_SIZE_K > 16:
        # print(K, SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT)
        # print(K % (SPLITK_BLOCK_SIZE // 2) == 0, SPLITK_BLOCK_SIZE % BLOCK_SIZE_K == 0, K % (BLOCK_SIZE_K // 2) == 0)

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

    # print(K, SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2, NUM_KSPLIT)
    # print(K % (SPLITK_BLOCK_SIZE // 2) == 0, SPLITK_BLOCK_SIZE % BLOCK_SIZE_K == 0, K % (BLOCK_SIZE_K // 2) == 0)
    return SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT


# Wrapper for gemm kernel.
def gemm_a8wfp4(
    x,
    w,
    y,
    x_scales,
    w_scales,
    dtype: Optional[float] = torch.bfloat16,
):
    """
    Computes the matmul Y = X @ W.T (where W.T is the logical transpose of unpacked W)
    
    X is in fp8 e4m3 format.
    W is in packed microscale fp4 (mxfp4) format, where 2 fp4 values are packed per uint8.
    x_scales are in fp32 format (one scale per row of X).
    w_scales are in e8m0 format (one scale per group of 32 elements in K dimension).

    Key parameters:
    - x: Matrix X with shape (M, K) in fp8 e4m3 format
    - w: Matrix W with shape (K//2, N) in packed fp4 format (2 values per uint8)
    - y: Pre-allocated output matrix with shape (M, N)
    - x_scales: Per-row scales for X with shape (M, 1) in fp32 format
    - w_scales: Per-group scales for W with shape (N, K//32) in e8m0 format
    - dtype: Output data type (default: torch.bfloat16)

    Returns:
    - y: The output matrix with shape (M, N) containing X @ W.T

    Note: 
    - W is stored in packed format where each uint8 contains 2 fp4 values
    - The logical shape of W after unpacking would be (K, N)
    - Every 32 consecutive elements in the K dimension of W share one e8m0 scale
    - X uses per-row scaling (not per-group scaling)
    """

    M, K = x.shape
    K_packed, N = w.shape
    assert K_packed == K // 2, f"Inconsistent shapes: x has K={K} but w has K_packed={K_packed}, expected {K//2}"
    assert x_scales.shape[0] == M and w_scales.shape == (N, K // 32), \
        f"Scale shapes incorrect: x_scales should be ({M}, 1), got {x_scales.shape}; w_scales should be ({N}, {K//32}), got {w_scales.shape}"

    if M < 32:
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 256
        GROUP_SIZE_M = 1
        waves_per_eu = 6
        kpack = 1
        num_warps = 4
        num_stages = 2
        matrix_instr_nonkdim = 16
        cache_modifier = ".cg"

        NUM_KSPLIT = 4
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, BLOCK_SIZE_K, NUM_KSPLIT
        )

        if os.getenv("VLLM_TRITON_FP4_GEMM_SPLITK_USE_BF16") == "1":
            y_pp = torch.empty((NUM_KSPLIT, M, N), dtype=y.dtype, device=y.device)
        else:
            y_pp = torch.empty((NUM_KSPLIT, M, N), dtype=torch.float32, device=y.device)
    elif M <= 128:
        BLOCK_SIZE_M = triton.next_power_of_2(M)
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 256
        GROUP_SIZE_M = 1
        waves_per_eu = 4
        kpack = 1
        num_warps = 4
        num_stages = 2
        matrix_instr_nonkdim = 16
        cache_modifier = ".cg"

        NUM_KSPLIT = 4
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, BLOCK_SIZE_K, NUM_KSPLIT
        )

        if os.getenv("VLLM_TRITON_FP4_GEMM_SPLITK_USE_BF16") == "1":
            y_pp = torch.empty((NUM_KSPLIT, M, N), dtype=y.dtype, device=y.device)
        else:
            y_pp = torch.empty((NUM_KSPLIT, M, N), dtype=torch.float32, device=y.device)
    elif M <= 256:
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 256
        GROUP_SIZE_M = 2
        waves_per_eu = 4
        kpack = 1
        num_warps = 4
        num_stages = 2
        matrix_instr_nonkdim = 16
        cache_modifier = ".cg"

        NUM_KSPLIT = 1
        SPLITK_BLOCK_SIZE = 2 * K
        y_pp = None
    else:
        BLOCK_SIZE_M = 256
        BLOCK_SIZE_N = 256
        BLOCK_SIZE_K = 256
        GROUP_SIZE_M = 32
        waves_per_eu = 1
        kpack = 1
        num_warps = 8
        num_stages = 2
        matrix_instr_nonkdim = 32
        cache_modifier = None

        NUM_KSPLIT = 1
        SPLITK_BLOCK_SIZE = 2 * K
        y_pp = None

    grid = lambda META: (  # noqa: E731
        (
            NUM_KSPLIT
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )

    y_final =  y if NUM_KSPLIT == 1 else y_pp
    stride_am, stride_ak = x.stride()
    stride_bk, stride_bn = w.stride()
    stride_ck, stride_cm, stride_cn = (0, y.stride(0), y.stride(1)) if NUM_KSPLIT == 1 else y_pp.stride()
    stride_asm, stride_ask = x_scales.stride()
    stride_bsn, stride_bsk = w_scales.stride()

    if DEBUG:
        print("grid:", grid({"BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N}) )
        print("x:", x)
        print("w:", w)
        print("y_final:", y_final)
        print("x_scales", x_scales)
        print("w_scales:", w_scales)
        print("M:", M)
        print("N:", N)
        print("K:", K)
        print("stride_am:", stride_am)
        print("stride_ak:", stride_ak)
        print("stride_bk:", stride_bk)
        print("stride_bn:", stride_bn)
        print("stride_ck:", stride_ck)
        print("stride_cm:", stride_cm)
        print("stride_cn:", stride_cn)
        print("stride_asm:", stride_asm)
        print("stride_ask:", stride_ask)
        print("stride_bsn:", stride_bsn)
        print("stride_bsk:", stride_bsk)
        print("BLOCK_SIZE_M:", BLOCK_SIZE_M)
        print("BLOCK_SIZE_N:", BLOCK_SIZE_N)
        print("BLOCK_SIZE_K:", BLOCK_SIZE_K)
        print("GROUP_SIZE_M:", GROUP_SIZE_M)
        print("NUM_KSPLIT:", NUM_KSPLIT)
        print("SPLITK_BLOCK_SIZE:", SPLITK_BLOCK_SIZE)
        print("cache_modifier:", cache_modifier)

    _gemm_a8wfp4_kernel[grid](
        x,
        w,
        y_final,
        x_scales,
        w_scales,
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
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        NUM_KSPLIT,
        SPLITK_BLOCK_SIZE,
        RAW_MASKED_LOADS=True,
        cache_modifier=cache_modifier,
        waves_per_eu=waves_per_eu,
        kpack=kpack,
        num_warps=num_warps,
        num_stages=num_stages,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
    )

    if NUM_KSPLIT > 1:
        REDUCE_BLOCK_SIZE_M = 16
        # TODO: Need to debug - REDUCE_BLOCK_SIZE_N=128 with fp32 partials fails
        # NOTE: REDUCE_BLOCK_SIZE_N=16 gives best perf with fp32 partials and
        # REDUCE_BLOCK_SIZE_N=128 gives best perf with bf16 partials
        REDUCE_BLOCK_SIZE_N = (
            128 if os.getenv("VLLM_TRITON_FP4_GEMM_SPLITK_USE_BF16") == "1" else 64
        )
        ACTUAL_KSPLIT = triton.cdiv(K, (SPLITK_BLOCK_SIZE))

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_afp4_wfp4_reduce_kernel[grid_reduce](
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
            NUM_KSPLIT,
        )