import triton
import triton.language as tl


@triton.jit
def _swiglu(
    out_ptr,
    x_ptr,
    x_stride,
    out_stride,
    n_rows,
    n_cols,
    B0: tl.constexpr,
):
    # break in half. apply swish to one half. multiply with the second half
    # for simplicity, we assume the same matrix size, similar to the process for FC layer.
    # so the same matrix will be used in both cases, one copy will have swish applied, and that will be (element-wise) multiplied to the original
    # in the spirit of FC1 layer :)
    pid = tl.program_id(0)
    start = x_ptr + pid * x_stride
    for col in range(0, n_cols, B0):  # loop down columns to load in the row. could maybe do this with another axis?
        offsets = tl.arange(0, B0) + col  # ?
        total_offsets = offsets + start
        mask = offsets < n_cols
        # load row
        row = tl.load(total_offsets, mask=mask, other=-float("inf"))
        # do a swish on the row -- x2 * sigmoid(x2) -- sigmoid = 1/(1+e^(-x))
        # this is like torch
        if row.dtype == tl.float16 or row.dtype == tl.bfloat16:
            row = tl.cast(row, tl.float32)
        #tl.device_print(row.dtype == tl.float16)
        # sigmoid = 1 / (1 + tl.exp(-row))
        swiglu = row * tl.sigmoid(row) * row  # element-wise multiplication, pretty sure you can do the * operator in triton? says in the docs magic operators are implemented
        # store the result
        out_row = out_ptr + pid * x_stride  # find the right row
        out = out_row + offsets
        tl.store(out, swiglu, mask=mask)
