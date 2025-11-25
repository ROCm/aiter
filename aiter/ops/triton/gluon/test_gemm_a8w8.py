import torch
import torch.nn.functional as F

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.test_common import perftest, checkAllclose

BLOCK_SIZE_M = 128
BLOCK_SIZE_K = 128
BLOCK_SIZE_N = 128
GROUP_SIZE_M = 4

in_dtype = torch.float8_e4m3fn
out_dtype = torch.bfloat16

def generate_a8w8_inputs(
    M: int,
    N: int,
    K: int,
):
    x = torch.randn((M, K), dtype=torch.float32, device="cuda")
    weight = torch.randn((N, K), dtype=torch.float32, device="cuda")
    max_x = x.abs().float().amax(dim=1, keepdim=True)
    x_scale = max_x / torch.finfo(in_dtype).max
    x = x / x_scale
    x = x.to(in_dtype)
    
    max_weight = weight.abs().float().amax(dim=1, keepdim=True).T.contiguous()
    w_scale = max_weight / torch.finfo(in_dtype).max
    weight = weight / w_scale.T
    weight = weight.to(in_dtype)
    return x, weight, x_scale, w_scale


def run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    x = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    return out.to(dtype)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _triton_gemm_a8w8_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_a_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
    offs_b_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
    a_scale = tl.load(a_scale_ptr + offs_a_scale)
    b_scale = tl.load(b_scale_ptr + offs_b_scale)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator += tl.dot(a, b, input_precision="ieee")

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator *= a_scale[:, None] * b_scale[None, :]
    c = accumulator.to(c_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@perftest()
def run_triton(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    assert x.shape[1] == weight.shape[1]
    M, K = x.shape
    N, K = weight.shape

    weight = weight.T

    y = torch.empty((M, N), dtype=dtype, device=x.device)

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )
    _triton_gemm_a8w8_kernel[grid](
        x,
        weight,
        x_scale,
        w_scale,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        NUM_XCDS=8,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return y
    
    
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit
def _gluon_gemm_a8w8_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    cache_modifier: gl.constexpr,
    cdna_version: gl.constexpr,
):
    pid = gl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    threads_per_elem_mk: gl.constexpr = triton.cdiv(
        BLOCK_SIZE_M * BLOCK_SIZE_K // (NUM_WARPS * 64), 16
    )

    threads_per_elem_kn: gl.constexpr = triton.cdiv(
        BLOCK_SIZE_K * BLOCK_SIZE_N // (NUM_WARPS * 64), 16
    )

    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[threads_per_elem_mk, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[0, 1],
    )

    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[threads_per_elem_kn, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[0, 1],
    )
    
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=cdna_version,
        instr_shape=[16, 16],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0, 1]
    )
    shared_a_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )
    shared_b_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )

    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_mk))
    offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_kn))

    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )
    smem_scale_a = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty, [BLOCK_SIZE_M], layout=shared_a_scale
    )
    smem_scale_b = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [BLOCK_SIZE_N], layout=shared_b_scale
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    offs_am = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk)
    )
    offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn)
    )

    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    offs_a_scale = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk)
    )
    offs_b_scale = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn)
    )
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=a_scale_ptr,
        offsets=offs_a_scale,
        mask=offs_am < M,
        cache=cache_modifier,
    )
    b_scale = gl.amd.cdna4.buffer_load(
        ptr=b_scale_ptr,
        offsets=offs_b_scale,
        mask=offs_bn < N,
        cache=cache_modifier,
    )
    smem_scale_a.store(a_scale)
    smem_scale_b.store(b_scale)

    
    acc = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout
    )
    zeros = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout
    )
    
    for k in range(0, gl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = gl.amd.cdna4.buffer_load(
                ptr=a_ptr,
                offsets=offs_a,
                mask=offs_am[:, None] < M,
                cache=cache_modifier,
            )
            b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
                mask=offs_bn[None, :] < N,
                cache=cache_modifier,
            )
        else:
            a = gl.amd.cdna4.buffer_load(
                ptr=a_ptr,
                offsets=offs_a,
                mask=(offs_am[:, None] < M) & (offs_ak[None, :] < K - k * BLOCK_SIZE_K),
                cache=cache_modifier,
            )
            b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
                mask=(offs_bn[None, :] < N) & (offs_bk[:, None] < K - k * BLOCK_SIZE_K),
                cache=cache_modifier,
            )
        smem_a.store(a)
        smem_b.store(b)
        
        cur_a = smem_a.load(layout=dot_a_layout)
        cur_b = smem_b.load(layout=dot_b_layout)
        
        mfma_out = gl.amd.cdna4.mfma(cur_a, cur_b, zeros)
        acc += mfma_out
        
        offs_a += BLOCK_SIZE_K * stride_ak
        offs_b += BLOCK_SIZE_K * stride_bk
        
    cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, mfma_layout))
    cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, mfma_layout))
    acc = acc * cur_a_scale[:, None] * cur_b_scale[None, :]
        
    c = acc.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    gl.amd.cdna4.buffer_store(
        stored_value=c, ptr=c_ptr, offsets=offs_c, mask=mask_c
    )
    

@perftest()
def run_gluon(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    assert x.shape[1] == weight.shape[1]
    M, K = x.shape
    N, K = weight.shape

    weight = weight.T

    y = torch.empty((M, N), dtype=dtype, device=x.device)

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )
    _gluon_gemm_a8w8_kernel[grid](
        x,
        weight,
        x_scale,
        w_scale,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        NUM_XCDS=8,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_WARPS=4,
        cache_modifier=".cg",
        cdna_version=3,
    )
    return y
    

def test_gemm(m, n, k):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    x, weight, x_scale, w_scale = generate_a8w8_inputs(m, n, k)

    ref = run_torch(x, weight, x_scale, w_scale)
    # triton_out, triton_us = run_triton(x, weight, x_scale, w_scale)
    gluon_out, gluon_us = run_gluon(x, weight, x_scale, w_scale)
    # checkAllclose(ref, triton_out, rtol=1e-2, atol=1e-2)
    checkAllclose(ref, gluon_out, rtol=1e-2, atol=1e-2)
    # print(f'[DEBUG] triton_us: {triton_us:.2f}us, gluon_us: {gluon_us:.2f}us')


if __name__ == "__main__":
    test_gemm(1024, 1024, 1024)
