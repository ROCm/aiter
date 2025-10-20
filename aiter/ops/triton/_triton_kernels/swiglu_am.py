import triton
import triton.language as tl


@triton.jit
def _swiglu(
    out_ptr,
    input_ptr, # token embeddings x hidden size
    layer_ptr, # hidden size x intermediate dim
    a_rows_stride, 
    a_cols_stride,
    b_rows_stride,
    b_cols_stride, 
    out_rows_stride,
    out_cols_stride,
    token_embeddings, # num rows of A
    hidden_size, # num cols of A, num rows of B
    intermediate_dim, # num cols of B
    B0: tl.constexpr, # one for general rows and such
    B1: tl.constexpr, # for cols for B
    BK: tl.constexpr, # for hidden size
):
    # break in half. apply swish to one half. multiply with the second half
    # for simplicity, we assume the same matrix size, similar to the process for FC layer.
    # so the same matrix will be used in both cases, one copy will have swish applied, and that will be (element-wise) multiplied to the original
    # in the spirit of FC1 layer :)

    # the assignment has changed. do GEMM like f1 layer, then for every row calculated, have one copy of original and save one swiglu'd version. 
    # then multiply together and store the row
    # assuming two layers so ends up with Mx2N as the output, then multiply element-wise together

    pid = tl.program_id(0)
    offsets_a = pid * B0 + tl.arange(0, B0)
    offsets_b = pid * B1 + tl.arange(0, B1)
    offsets_k = tl.arange(0, BK)
    a_ptrs = input_ptr + (offsets_a * a_rows_stride + offsets_k * a_cols_stride)
    b_ptrs = layer_ptr + (offsets_k * b_rows_stride + offsets_b * b_cols_stride)

    # accumulator for fp16->fp32 ala gemm tutorial
    out_setup = tl.zeros((B0, B1), dtype=tl.float32)
    for k in range(0, tl.cdiv(hidden_size, BK)):
        mask = offsets_k < hidden_size - k * BK
        a = tl.load(a_ptrs, mask=mask)
        b = tl.load(b_ptrs, mask=mask)
        # acc along k, then go to next k block assumably if my code is right
        row = tl.dot(a, b, out_setup)
        # activation function with acc -- swiglu
        out_setup = row * tl.sigmoid(row) * row # hopefully this is just the row and not the whole acc
        a_ptrs += BK * a_cols_stride
        b_ptrs += BK * b_rows_stride
    swiglu_final = out_setup.to(tl.float16)

    offsets_out_rows = pid * B0 + tl.arange(0, B0)
    offsets_out_cols = pid * B1 + tl.arange(0, B1)
    out_ptrs = out_ptr + out_rows_stride * offsets_out_rows + out_cols_stride * offsets_out_cols
    out_mask = (offsets_out_rows < token_embeddings) & (offsets_out_cols < intermediate_dim)
    tl.store(out_ptrs, swiglu_final, mask=out_mask)
