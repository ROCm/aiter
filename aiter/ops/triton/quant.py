import triton
import triton.language as tl
import torch



@triton.jit
def _per_tensor_fp8_quant(x_in: torch.Tensor, 
                                scale_in: torch.Tensor
): 
    """
    Quantize to fp8 with the scale provided

    Parameters:
    x_in: Tensor to be quantized
    scale_in: Scale value of shape [1]. Used for all the elements in the x_in

    Returns:
    qx: quantized values. Must have same shape as x_in

    Note: x_in, scale_in must be loaded in the caller triton jitted function
    """
    qx = x_in / scale_in
    qx = qx.to(tl.float8e4b8)

    return qx

@triton.jit
def _static_per_tensor_fp8_quant_kernel(qx_ptr: torch.Tensor, 
                                        x_in_ptr: torch.Tensor, 
                                        scale_in_ptr: torch.Tensor,
                                        rows: int,
                                        cols: int,
                                        qx_stride_r: int,
                                        qx_stride_c: int,
                                        x_in_stride_r: int,
                                        x_in_stride_c: int,
                                        BLOCK_SIZE: tl.constexpr,
                                        NUM_COL_POW2: tl.constexpr
):
    """
    #TODO: Add Doc
    """

    pid = tl.program_id(axis=0)

    offs = pid*BLOCK_SIZE*qx_stride_r + tl.arange(0, BLOCK_SIZE)[:, None]*qx_stride_r + tl.arange(0, NUM_COL_POW2)[None, :]
    mask = (tl.arange(0, BLOCK_SIZE) < rows)[:, None] &  (tl.arange(0, NUM_COL_POW2) < cols)[None, :]   
    x = tl.load(x_in_ptr + offs, mask = mask)

    scale = tl.load(scale_in_ptr)

    qx = _per_tensor_fp8_quant(x, scale)

    tl.store(qx_ptr + offs, qx, mask=mask)


def static_per_tensor_fp8_quant(qx: torch.Tensor, 
                                x_in: torch.Tensor, 
                                scale_in: torch.Tensor
):

    """
    #TODO: Add Doc
    """
    assert scale_in.numel() == 1 #only single scale value 
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    BLOCK_SIZE = 32
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (triton.cdiv(rows,meta['BLOCK_SIZE']),)  
    _static_per_tensor_fp8_quant_kernel[grid](qx, 
                                            x_in, 
                                            scale_in, 
                                            rows, 
                                            cols, 
                                            qx.stride(0), 
                                            qx.stride(1), 
                                            x_in.stride(0),
                                            x_in.stride(1),
                                            BLOCK_SIZE=BLOCK_SIZE,
                                            NUM_COL_POW2=NUM_COL_POW2
                                            )

    return qx



@triton.jit
def _dynamic_per_tensor_fp8_quant_kernel(qx_ptr: torch.Tensor, 
                                        scale_out_ptr: torch.Tensor,
                                        x_max_ptr: torch.Tensor,
                                        x_in_ptr: torch.Tensor, 
                                        rows: int,
                                        cols: int,
                                        qx_stride_r: int,
                                        qx_stride_c: int,
                                        x_in_stride_r: int,
                                        x_in_stride_c: int,
                                        BLOCK_SIZE: tl.constexpr,
                                        NUM_COL_POW2: tl.constexpr,
                                        FP8_MAX: tl.constexpr
):
    """
    #TODO: Add Doc
    """

    pid = tl.program_id(axis=0)

    offs = pid*BLOCK_SIZE*x_in_stride_r + tl.arange(0, BLOCK_SIZE)[:, None]*x_in_stride_r + tl.arange(0, NUM_COL_POW2)[None, :]
    mask = (pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < rows)[:, None] &  (tl.arange(0, NUM_COL_POW2) < cols)[None, :]   
    x = tl.load(x_in_ptr + offs, mask = mask)

    m = tl.max(tl.abs(x))
    tl.atomic_max(x_max_ptr, m)
    x_max = tl.load(x_max_ptr)
    
    scale_out = x_max / FP8_MAX
    
    qx = _per_tensor_fp8_quant(x, scale_out)

    tl.store(qx_ptr + offs, qx, mask=mask)
    tl.store(scale_out_ptr, scale_out)



def dynamic_per_tensor_fp8_quant(qx: torch.Tensor, 
                                x_in: torch.Tensor,
                                scale_out: torch.Tensor
):

    """
    #TODO: Add Doc
    """
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    x_max = torch.zeros((1), dtype=torch.float32, device='cuda')
    BLOCK_SIZE = 32
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (triton.cdiv(rows,meta['BLOCK_SIZE']),)  
    _dynamic_per_tensor_fp8_quant_kernel[grid](qx, 
                                            scale_out,
                                            x_max,
                                            x_in, 
                                            rows, 
                                            cols, 
                                            qx.stride(0), 
                                            qx.stride(1), 
                                            x_in.stride(0),
                                            x_in.stride(1),
                                            BLOCK_SIZE=BLOCK_SIZE,
                                            NUM_COL_POW2=NUM_COL_POW2,
                                            FP8_MAX=torch.finfo(torch.float8_e4m3fnuz).max
                                            )

    return qx, scale_out

@triton.jit
def _dynamic_per_token_fp8_quant_kernel(qx_ptr: torch.Tensor,
                                        scale_out_ptr: torch.Tensor,
                                        x_in_ptr: torch.Tensor,
                                        rows: int,
                                        cols: int,
                                        qx_stride_r: int,
                                        qx_stride_c: int,
                                        x_in_stride_r: int,
                                        x_in_stride_c: int,
                                        BLOCK_SIZE: tl.constexpr,
                                        NUM_COL_POW2: tl.constexpr,
                                        FP8_MAX: tl.constexpr
):
    """
    #TODO: Add Doc
    """

    pid = tl.program_id(axis=0)

    offs = pid*BLOCK_SIZE*x_in_stride_r + tl.arange(0, BLOCK_SIZE)[:, None]*x_in_stride_r + tl.arange(0, NUM_COL_POW2)[None, :]
    mask = (pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < rows)[:, None] &  (tl.arange(0, NUM_COL_POW2) < cols)[None, :]   
    x = tl.load(x_in_ptr + offs, mask = mask)

    m = tl.max(tl.abs(x), axis=-1)
    scale_out = m / FP8_MAX

    qx = x / scale_out[:, None]
    qx = qx.to(tl.float8e4b8)

    tl.store(qx_ptr + offs, qx, mask=mask)

    scale_offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    scale_mask = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < rows
    tl.store(scale_out_ptr + scale_offs, scale_out, mask=scale_mask)


def dynamic_per_token_fp8_quant(qx: torch.Tensor,
                                x_in: torch.Tensor,
                                scale_out: torch.Tensor
):

    """
    #TODO: Add doc
    """
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    BLOCK_SIZE = 32
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (triton.cdiv(rows,meta['BLOCK_SIZE']),)  
    _dynamic_per_token_fp8_quant_kernel[grid](qx, 
                                            scale_out,
                                            x_in, 
                                            rows, 
                                            cols, 
                                            qx.stride(0), 
                                            qx.stride(1), 
                                            x_in.stride(0),
                                            x_in.stride(1),
                                            BLOCK_SIZE=BLOCK_SIZE,
                                            NUM_COL_POW2=NUM_COL_POW2,
                                            FP8_MAX=torch.finfo(torch.float8_e4m3fnuz).max
                                            )

    return qx, scale_out

   



