import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(
    output_ptr,
    input_ptr,
    rows,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    curRow = input_ptr + pid
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < cols

    row_ptr = input_ptr + pid * cols + offset
    x = tl.load(row_ptr, mask=mask)
    
    # do softmax 
    zMax = tl.max(x,0)
    numerator = tl.exp(x - zMax)
    denomenator = tl.sum(numerator, 0)
    softmax_out = numerator / denomenator

    out_ptr = output_ptr + pid * cols + offset 
    tl.store(out_ptr, softmax_out,mask=mask)