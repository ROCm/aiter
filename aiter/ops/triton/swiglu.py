import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.swiglu import _swiglu

def swiglu(x, W, V, b, c, BLOCK_SIZE_M, BLOCK_SIZE_N):
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
    M, K = x.shape
    K, N = W.shape
    out = torch.empty((M, N), dtype=torch.float32)
    
    # Check constraints.
    assert x.shape[1] == W.shape[0], "Incompatible dimensions!!!"

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M), 
        triton.cdiv(N, BLOCK_SIZE_N)
    )

    res = _swiglu[grid](
        out,
        x,
        W, V, b, c,
        M, K, N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    return res

