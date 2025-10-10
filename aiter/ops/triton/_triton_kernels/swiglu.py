import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    output_ptr,          # [n_rows, n_cols]
    a_ptr,               # [n_rows, n_cols]   (first half of FC projection)
    b_ptr,               # [n_rows, n_cols]   (second half of FC projection)
    row_stride_out,      # stride (in elements) between consecutive rows in output
    row_stride_a,        # stride for A
    row_stride_b,        # stride for B
    n_rows,              # number of rows (M)
    n_cols,              # number of columns (N)
    BLOCK_SIZE: tl.constexpr,  # tile size along columns (threads per program)
):
    # one program per row (axis 0), identical “grid = (n_rows,)” launch pattern to your softmax
    row_id = tl.program_id(0)  # program(instruction) row index, one program per row along the "0 axis"

    # find ptr address of the start of the row for each tensor
    row_out_ptr = output_ptr + row_id * row_stride_out
    row_a_ptr   = a_ptr      + row_id * row_stride_a
    row_b_ptr   = b_ptr      + row_id * row_stride_b

    block_offset = tl.arange(0, BLOCK_SIZE)  # size of block that threads will be working on

    # iterate across the row in tiles of width BLOCK_SIZE
    col_start = 0
    while col_start < n_cols:

        col = col_start + block_offset  # getting the actual indicies of each of the elements in the tile

        valid = col < n_cols  # used to ensure that the index is valid or needs to be masked

        # load A and B tiles; for invalid lanes, “other” won’t be used thanks to mask on store
        a_tile = tl.load(row_a_ptr + col, mask=valid, other=0.0)
        b_tile = tl.load(row_b_ptr + col, mask=valid, other=0.0)

        # compute SiLU(b) = b * sigmoid(b) in fp32 for stability, then cast back
        a32 = a_tile.to(tl.float32)  # in case incoming is fp16/bf16
        b32 = b_tile.to(tl.float32)

        # sigmoid(x) = 1 / (1 + exp(-x))
        neg_b = -b32
        exp_nb = tl.exp(neg_b)
        sig_b = 1.0 / (1.0 + exp_nb)

        silu_b = b32 * sig_b  # SiLU(b)
        y32 = a32 * silu_b    # A * SiLU(B)

        y = y32.to(a_tile.dtype)  # go back to original dtype (fp16/bf16)

        tl.store(row_out_ptr + col, y, mask=valid)  # store the tile

        col_start = col_start + BLOCK_SIZE  # next tile
