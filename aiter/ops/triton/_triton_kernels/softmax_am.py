import triton
import triton.language as tl


@triton.jit
def _softmax(
    out_ptr,
    x_ptr,
    x_stride,
    out_stride,
    n_rows,
    n_cols,
    B0: tl.constexpr,
):
    # softmax to 2d shape (matrix). 
    # applying softmax to each row -> softmax vals for each col
    # k classes, assume k to rep rows. so loop over each row, load col, softmax each row to each col
    # im still thinking in cuda!! no!!
    # one pid. each row goes in one block.
    pid = tl.program_id(0)
    row_start = x_ptr + pid * x_stride
    
    # from the beginning of the execution, you are dealing with processing a row.
    # the loop is dealing with strafing along the row to load each block.
    # this looks like it's looping in rows. it is not. each row is executing in parallel

    row_max = -float('inf')
    row_sum = 0.0

    # find max in row by strafing along blocks. sum each block as you go
    for col in range(0, n_cols, B0):  # looping across rows to process each element. we use triton blocks across columns. 
                               # we load row in to do calculations
                    # note: this is not looping across rows. this is looping across blocks along the number of columns. 1 block = 1 pass
        offsets = tl.arange(0, B0) + col
        total_offsets = row_start + offsets
        mask = offsets < n_cols
        # load block in
        current_block = tl.load(total_offsets, mask=mask, other=-float("inf"))
        # find max in row --- potential issue found, forgot to account for max in block size. TODO
        new_max = max(row_max, tl.max(current_block, axis=0))
        # it's meant to be the sum of the exponents, not the flat sum of the row!
        row_sum = row_sum * tl.exp(row_max - new_max)
        row_sum += tl.sum(tl.exp(current_block - new_max))
        row_max = new_max
        # move on

    # perform operations on the actual row and store results
    # strafe along blocks
    for col in range(0, n_cols, B0):  # looping across rows to process each element. we use triton blocks across columns. 
                               # we load row in to do calculations
                    # note: this is not looping across rows. this is looping across blocks along the number of columns. 1 block = 1 pass
        offsets = tl.arange(0, B0) + col
        total_offsets = row_start + offsets
        mask = offsets < n_cols
        # load block in
        current_block = tl.load(total_offsets, mask=mask, other=-float("inf"))
        # softmax row
        num = tl.exp(current_block - row_max)
        softmax_total = num / row_sum
        # store block of new row
        out_row_start = out_ptr + pid * x_stride
        out = out_row_start + offsets
        tl.store(out, softmax_total, mask=mask)
        # keep going until all blocks processed