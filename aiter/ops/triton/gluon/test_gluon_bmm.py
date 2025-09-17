import torch
import triton
import triton.language as tl
import triton.testing as tt
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import triton.experimental.gluon.language as ttgl
import triton.experimental.gluon.language.amd as amd_layouts  

def get_autotune_config():
    sizes = [
        {'BLOCK_B': 16, 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64},
    ]
    return [triton.Config(s) for s in sizes]

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@gluon.jit
def matmul_3d_kernel(  
    a_ptr, b_ptr, c_ptr,  
    batch_size, M, N, K,  
    stride_a_batch, stride_a_m, stride_a_k,  
    stride_b_batch, stride_b_k, stride_b_n,  
    stride_c_batch, stride_c_m, stride_c_n,  
    BLOCK_B: ttgl.constexpr,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr  
):
    mfma_layout: ttgl.constexpr = amd_layouts.AMDMFMALayout(  
        version=3,   
        instr_shape=[16, 16],
        transposed=False,   
        warps_per_cta=[4, 1, 1]     # [batch, M, N]
        # warps_per_cta=[2, 2]     # [batch, M, N]
    )

    dot_a_layout: ttgl.constexpr = ttgl.DotOperandLayout(  
        operand_index=0,   
        parent=mfma_layout,   
        k_width=16  
    )  
    dot_b_layout: ttgl.constexpr = ttgl.DotOperandLayout(  
        operand_index=1,   
        parent=mfma_layout,   
        k_width=16  
    )

    blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(  
        # size_per_thread=[1, 8, 8],
        # threads_per_warp=[1, 8, 8],
        # warps_per_cta=[4, 1, 1],
        size_per_thread=[1, 4, 4],
        threads_per_warp=[1, 8, 8],
        warps_per_cta=[1, 2, 2],
        # order=[0, 1, 2],
        order=[2, 1, 0],
    )
    # blocked_layout: ttgl.constexpr = ttgl.BlockedLayout(  
    #     size_per_thread=[4, 4],
    #     threads_per_warp=[8, 8],  
    #     warps_per_cta=[2, 2],  
    #     order=[1, 0],
    # )

    pid = gl.program_id(axis=0)  
    pid_batch = pid // (gl.cdiv(M, BLOCK_M) * gl.cdiv(N, BLOCK_N))
    pid_mn = pid % (gl.cdiv(M, BLOCK_M) * gl.cdiv(N, BLOCK_N))
    pid_m = pid_mn // gl.cdiv(N, BLOCK_N)
    pid_n = pid_mn % gl.cdiv(N, BLOCK_N)

    offs_batch = gl.arange(0, BLOCK_B, layout=ttgl.SliceLayout(1, ttgl.SliceLayout(2, blocked_layout)))
    offs_batch2 = gl.arange(0, 1, layout=ttgl.SliceLayout(1, ttgl.SliceLayout(2, blocked_layout)))
    offs_am = gl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(0, ttgl.SliceLayout(2, blocked_layout)))
    offs_bn = gl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, ttgl.SliceLayout(1, blocked_layout)))
    offs_ak = gl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, ttgl.SliceLayout(1, blocked_layout)))
    offs_bk = gl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, ttgl.SliceLayout(2, blocked_layout)))
    # offs_am = gl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, blocked_layout))
    # offs_bn = gl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, blocked_layout))
    # offs_ak = gl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, blocked_layout))
    # offs_bk = gl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(1, blocked_layout))

    batch_offset = pid_batch * BLOCK_B
    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N

    offs_a = (batch_offset + offs_batch2)[:, None, None] * stride_a_batch + \
        (m_offset + offs_am)[None, :, None] * stride_a_m + \
        offs_ak[None, None, :] * stride_a_k
    offs_b = (batch_offset + offs_batch)[:, None, None] * stride_b_batch + \
        offs_bk[None, :, None] * stride_b_k + \
        (n_offset + offs_bn)[None, None, :] * stride_b_n
    mask_b = (batch_offset + offs_batch)[:, None, None] < batch_size
    accumulator = ttgl.zeros([BLOCK_B, BLOCK_M, BLOCK_N], ttgl.float32, mfma_layout)  
    # offs_a = batch_offset * stride_a_batch + (m_offset + offs_am)[:, None] * stride_a_m + offs_ak[None, :] * stride_a_k
    # offs_b = batch_offset * stride_b_batch + offs_bk[:, None] * stride_b_k + (n_offset + offs_bn)[None, :] * stride_b_n
    # accumulator = ttgl.zeros([BLOCK_M, BLOCK_N], ttgl.float32, mfma_layout)  

    for k in range(0, gl.cdiv(K, BLOCK_K)):
        # a = ttgl.load(a_ptr + offs_a)
        # # b = ttgl.load(b_ptr + offs_b)
        # # a = ttgl.load(a_ptr + offs_a, mask=mask_b)
        # b = ttgl.load(b_ptr + offs_b, mask=mask_b)
        a = gl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a)
        b = gl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=mask_b)
        a_broadcasted = tl.broadcast_to(a, BLOCK_B, BLOCK_M, BLOCK_N)

        a_dot = ttgl.convert_layout(a_broadcasted, dot_a_layout)
        b_dot = ttgl.convert_layout(b, dot_b_layout)

        accumulator = gl.amd.cdna3.mfma(a_dot, b_dot, accumulator)  

        a_ptr += BLOCK_K * stride_a_k
        b_ptr += BLOCK_K * stride_b_k

    c = ttgl.convert_layout(accumulator, blocked_layout)
    c = c.to(c_ptr.dtype.element_ty)
    offs_c = (batch_offset + offs_batch)[:, None, None] * stride_c_batch + \
        (m_offset + offs_am)[None, :, None] * stride_c_m + \
        (n_offset + offs_bn)[None, None, :] * stride_c_n
    ttgl.store(c_ptr + offs_c, c, mask=mask_b)

    # c = accumulator.to(c_ptr.dtype.element_ty)
    # offs_cb = gl.arange(0, 1, layout=gl.SliceLayout(1, gl.SliceLayout(2, mfma_layout)))
    # offs_cm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, gl.SliceLayout(2, mfma_layout)))
    # offs_cn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, gl.SliceLayout(1, mfma_layout)))
    # offs_c = (batch_offset + offs_cb)[:, None, None] * stride_c_batch + \
    #     (m_offset + offs_cm)[None, :, None] * stride_c_m + \
    #     (n_offset + offs_cn)[None, None, :] * stride_c_n
    # c_mask = (offs_cm[None, :, None] < M) & (offs_cn[None, None, :] < N)
    # ttgl.amd.cdna3.buffer_store(stored_value=c, ptr=c_ptr, offsets=offs_c, mask=c_mask)

    # offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout))
    # offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    # offs_c = batch_offset * stride_c_batch + offs_cm[:, None] * stride_c_m + offs_cn[None, :] * stride_c_n
    # # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # # ttgl.amd.cdna3.buffer_store(stored_value=c, ptr=c_ptr, offsets=offs_c, mask=c_mask)
    # # ttgl.amd.cdna3.buffer_store(stored_value=c, ptr=c_ptr, offsets=offs_c)
    # ttgl.store(c_ptr + offs_c, c)


def test_matmul_3d_kernel():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Test parameters
    # batch_size = 1
    # batch_size = 10
    # batch_size = 11
    batch_size = 16
    # M, N, K = 128, 128, 64
    M, N, K = 128, 128, 128
    # dtype = torch.float32
    # dtype = torch.float16
    dtype = torch.bfloat16

    # Create random input tensors
    a = torch.randn((1, M, K), device='cuda', dtype=dtype)
    # a_broadcasted = a.expand(batch_size, M, K)
    a_broadcasted = a.repeat(batch_size, 1, 1)
    b = torch.randn((batch_size, K, N), device='cuda', dtype=dtype)

    # Allocate output tensor
    c = torch.empty((batch_size, M, N), device='cuda', dtype=dtype)

    # Compute reference result using PyTorch
    c_ref = torch.bmm(a_broadcasted, b)

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_B']) * triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    matmul_3d_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        batch_size=batch_size, M=M, N=N, K=K,
        stride_a_batch=a.stride(0), stride_a_m=a.stride(1), stride_a_k=a.stride(2),
        stride_b_batch=b.stride(0), stride_b_k=b.stride(1), stride_b_n=b.stride(2),
        stride_c_batch=c.stride(0), stride_c_m=c.stride(1), stride_c_n=c.stride(2),
    )

    # Compare results
    # torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(c, c_ref)
    print("\033[92mPASSED\033[0m")


# Run the test
if __name__ == "__main__":
    test_matmul_3d_kernel()
