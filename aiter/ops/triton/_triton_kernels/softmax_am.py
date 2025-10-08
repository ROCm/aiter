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
    
    for col in range(0, n_cols, B0):  # looping across rows to process each element. we use triton blocks across columns. 
                               # we load row in to do calculations
        offsets = tl.arange(0, B0) + col
        total_offsets = row_start + offsets
        mask = offsets < n_cols
        # load row in
        row = tl.load(total_offsets, mask=mask, other=-float("inf"))
        # find max in row
        max_row = tl.max(row, axis=0)
        # softmax row
        num = tl.exp(row - max_row)
        denom = tl.sum(num, axis=0)
        softmax_total = num / denom
        # store new row
        out_row_start = out_ptr + pid * x_stride
        out = out_row_start + offsets
        tl.store(out, softmax_total, mask=mask)
        # move on


# come back and edit this one