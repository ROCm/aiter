import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(
    output_ptr,
    input_ptr,
    rows,
    cols,
    stride_in_rows,
    stride_out_rows,
    BLOCK_SIZE: tl.constexpr,
):
    p_id = tl.program_id(0)
    row_ptr = input_ptr + p_id * stride_in_rows
    offCol = tl.arange(0,BLOCK_SIZE) #offset column
    mask = offCol < cols

    x = tl.load(row_ptr + offCol, mask=mask, other=-float('inf'))
    
    # do softmax 
    zMax = tl.max(x,axis=0)
    temp = x - zMax
    numerator = tl.exp(temp)
    denominator = tl.sum(numerator, 0)
    softmax_out = numerator / denominator

    out_ptr = output_ptr + p_id * stride_out_rows 
    tl.store(out_ptr + offCol, softmax_out,mask=mask)