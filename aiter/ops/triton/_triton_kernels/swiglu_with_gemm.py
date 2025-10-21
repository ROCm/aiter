import triton
import triton.language as tl


@triton.jit
def _swiglu_with_gemm_kernel(
    input_ptr,  # M x N
    weights_ptr,  # N x 2K
    output_ptr,  # M x K
    input_row_stride,
    input_col_stride,
    weights_row_stride,
    weights_col_stride,
    output_row_stride,
    output_col_stride,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    blocks_per_row = tl.cdiv(K, BLOCK_SIZE)

    # get row and column block index
    row_pid = pid // blocks_per_row
    col_pid = pid % blocks_per_row

    # get row and column ranges
    row_range = tl.arange(0, BLOCK_SIZE)[:, None]
    col_range = tl.arange(0, BLOCK_SIZE)[None, :]

    # perform GEMM
    x1 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    x2 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for block in tl.range(0, N, BLOCK_SIZE):
        # get input ranges
        input_row_range = row_range + row_pid * BLOCK_SIZE
        input_col_range = col_range + block

        # get input mask
        input_row_mask = input_row_range < M
        input_col_mask = input_col_range < N
        input_mask = input_row_mask & input_col_mask

        # get input
        input_row_offsets = input_row_range * input_row_stride
        input_col_offsets = input_col_range * input_col_stride
        input_offsets = input_row_offsets + input_col_offsets
        input_tile = tl.load(input_ptr + input_offsets, mask=input_mask, other=0)

        # get weights ranges
        weights_row_range = row_range + block
        weights_col_range = col_range + col_pid * BLOCK_SIZE

        # get weights mask
        weights_row_mask = weights_row_range < N
        weights_col_mask = weights_col_range < K
        weights_mask = weights_row_mask & weights_col_mask

        # get weights row offsets
        weights_row_offsets = weights_row_range * weights_row_stride

        # get weights for x1
        weights_x1_col_offsets = weights_col_range * weights_col_stride
        weights_x1_offsets = weights_row_offsets + weights_x1_col_offsets
        weights_x1_tile = tl.load(
            weights_ptr + weights_x1_offsets, mask=weights_mask, other=0
        )

        # get weights for x2
        weights_x2_col_offsets = (weights_col_range + K) * weights_col_stride
        weights_x2_offsets = weights_row_offsets + weights_x2_col_offsets
        weights_x2_tile = tl.load(
            weights_ptr + weights_x2_offsets, mask=weights_mask, other=0
        )

        # matrix multiply accumulate
        x1 = tl.dot(input_tile, weights_x1_tile, x1)
        x2 = tl.dot(input_tile, weights_x2_tile, x2)

    # compute swiglu
    y = (
        tl.where(x1 >= 0, 1 / (1 + tl.exp(-x1)), tl.exp(x1) / (1 + tl.exp(x1)))
        * x1
        * x2
    )

    # get output ranges
    output_row_range = row_range + row_pid * BLOCK_SIZE
    output_col_range = col_range + col_pid * BLOCK_SIZE

    # get output mask
    output_row_mask = output_row_range < M
    output_col_mask = output_col_range < K
    output_mask = output_row_mask & output_col_mask

    # store results
    output_row_offsets = output_row_range * output_row_stride
    output_col_offsets = output_col_range * output_col_stride
    output_offsets = output_row_offsets + output_col_offsets
    tl.store(output_ptr + output_offsets, y, mask=output_mask)
