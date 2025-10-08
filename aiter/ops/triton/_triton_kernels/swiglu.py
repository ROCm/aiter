# SiLU(x) = x * sigmoid(x), where sigmoid(x) = 1 / (1 + exp(-x))
# SwiGLU(x) = SiLU(x1) * x2, where x1 and x2 are halves of x


import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    input_col_stride,
    output_row_stride,
    output_col_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.cdiv(n_rows * n_cols // 2, BLOCK_SIZE * BLOCK_SIZE)

    # get row and column block index
    row_pid = pid // num_pid
    col_pid = pid % num_pid

    # get row and column ranges
    row_range = tl.arange(0, BLOCK_SIZE)[:, None]
    col_range = tl.arange(0, BLOCK_SIZE)[None, :]
    mid = n_cols // 2
    mask = (row_range + row_pid * BLOCK_SIZE < n_rows) & (
        col_range + col_pid * BLOCK_SIZE < mid
    )

    # get first half
    x1_offsets = (
        row_range * input_row_stride
        + row_pid * BLOCK_SIZE
        + col_range * input_col_stride
        + col_pid * BLOCK_SIZE
    )
    x1 = tl.load(input_ptr + x1_offsets, mask)

    # get second half
    x2_offsets = (
        row_range * input_row_stride
        + row_pid * BLOCK_SIZE
        + col_range * input_col_stride
        + col_pid * BLOCK_SIZE
        + mid * input_col_stride
    )
    x2 = tl.load(input_ptr + x2_offsets, mask)

    # compute swiglu
    result = tl.sigmoid(x1) * x1 * x2

    # store results
    y_offsets = (
        row_range * output_row_stride
        + row_pid * BLOCK_SIZE
        + col_range * output_col_stride
        + col_pid * BLOCK_SIZE
    )
    tl.store(output_ptr + y_offsets, result, mask)
