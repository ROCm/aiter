import triton
import triton.language as tl


@triton.jit
def _softmax_kernel_online(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):

    row_start = tl.program_id(0)
    row_idx = row_start

    # loop 1, find max and sum
    m = -float("inf")  # Initial value of max
    row_sum = 0.0
    row_start_ptr = input_ptr + row_idx * input_row_stride
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = b + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row_block = tl.load(
            input_ptrs, mask=mask, other=-float("inf"), cache_modifier=".cg"
        )  # load block
        m_p = tl.max(row_block, axis=0)  # find block max
        m_p = tl.maximum(m, m_p)  # Find new max across all blocks so far
        row_sum = row_sum * tl.exp(m - m_p)  # Adjust previous sum
        row_sum += tl.sum(
            tl.exp(row_block - m_p)
        )  # Add to exponentiated sum of this block
        m = m_p  # save max

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    # Loop 2
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = b + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row_block = tl.load(
            input_ptrs, mask=mask, other=-float("inf"), cache_modifier=".cg"
        )  # load block
        # subtract, exponentiate and divide by sum
        softmax_output = tl.exp(row_block - m) / row_sum
        # store
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
