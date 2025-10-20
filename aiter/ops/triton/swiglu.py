import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.swiglu import _swiglu

def swiglu(x, W, V, b, c, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):
    """
    Does swiglu activation on a 2D tensor

    Key parameters:
        x (torch.Tensor): A 2D input tensor.
        W (torch.Tensor): First trainable weight matrix.
        V (torch.Tensor): Second trainable weight matrix.
        b (torch.Tensor): First offset parameter to add.
        c (torch.Tensor): Second offset parameter to add.
        BLOCK_SIZE_M: Number of blocks for rows
        BLOCK_SIZE_N: Number of blocks for columns
    Returns:
        out pointer pointing to the activated output tensor
    """

    M, tempK = x.shape
    K, N = W.shape
    assert tempK == 2 * K 
    out = torch.empty((M, N), dtype=x.dtype,device="cuda")
    
    # Check constraints.
    assert x.shape[1]//2 == W.shape[0], "Incompatible dimensions!!!"

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M), 
        triton.cdiv(N, BLOCK_SIZE_N)
    )

    res = _swiglu[grid](
        out,x,
        W, V, b, c,
        x.stride(0),x.stride(1),
        out.stride(0),out.stride(1),
        W.stride(0),W.stride(1),
        V.stride(0),V.stride(1),
        M, K, N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    return out

#Reference for testing
def swiglu_ref(x, W, V, b, c):
    # x: [M, 2K], W, V: [K, N], b, c: [N]
    M, tempK = x.shape
    K = W.shape[0]
    N = W.shape[1]
    x1 = x[:, :K]
    x2 = x[:, K:]
    y1 = torch.matmul(x1, W) + b
    y2 = torch.matmul(x2, V) + c
    silu = y1 * torch.sigmoid(y1)
    return silu * y2

