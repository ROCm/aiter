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
    blocks_per_row = tl.cdiv(n_cols // 2, BLOCK_SIZE)

    # get row and column block index
    row_pid = pid // blocks_per_row
    col_pid = pid % blocks_per_row

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
        + row_pid * BLOCK_SIZE * input_row_stride
        + col_range * input_col_stride
        + col_pid * BLOCK_SIZE * input_col_stride
    )
    x1 = tl.load(input_ptr + x1_offsets, mask).to(tl.float32)

    # get second half
    x2_offsets = (
        row_range * input_row_stride
        + row_pid * BLOCK_SIZE * input_row_stride
        + (col_range + mid) * input_col_stride
        + col_pid * BLOCK_SIZE * input_col_stride
    )
    x2 = tl.load(input_ptr + x2_offsets, mask).to(tl.float32)

    # compute swiglu
    y = tl.sigmoid(x1) * x1 * x2

    # store results
    y_offsets = (
        row_range * output_row_stride
        + row_pid * BLOCK_SIZE * output_row_stride
        + col_range * output_col_stride
        + col_pid * BLOCK_SIZE * output_col_stride
    )
    tl.store(output_ptr + y_offsets, y, mask)
