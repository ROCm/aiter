import triton
import triton.language as tl


@triton.jit
def _softmax_v2_kernel(
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

    # find max and sum of row
    row_max = float("-inf")
    row_sum = 0.0
    for block in tl.range(0, n_cols, BLOCK_SIZE):
        row_offsets = pid
        col_offsets = block + tl.arange(0, BLOCK_SIZE)
        offsets = row_offsets * input_row_stride + col_offsets * input_col_stride

        # load part of the row
        mask = col_offsets < n_cols
        partial_row = tl.load(input_ptr + offsets, mask=mask, other=float("-inf")).to(
            tl.float32
        )

        # adjust the running max and sum
        cur_max = tl.max(partial_row)
        if cur_max > row_max:
            # adjust sum if max is new
            row_sum *= tl.exp(row_max - cur_max)
            row_max = cur_max
        row_sum += tl.sum(tl.exp(partial_row - row_max))

    # perform softmax
    for block in tl.range(0, n_cols, BLOCK_SIZE):
        row_offsets = pid
        col_offsets = block + tl.arange(0, BLOCK_SIZE)
        offsets = row_offsets * output_row_stride + col_offsets * output_col_stride

        # load part of the row
        mask = col_offsets < n_cols
        partial_row = tl.load(input_ptr + offsets, mask=mask, other=float("-inf")).to(
            tl.float32
        )

        # compute partial softmax
        partial_softmax = tl.exp(partial_row - row_max) / row_sum
        tl.store(output_ptr + offsets, partial_softmax, mask=mask)
