import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

@triton.jit
def _softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    LOG2E: tl.constexpr = 1.4426950408889634
    # Rows of softmax are independent, so we parallelize across these rows
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    # Stride indicates how many elements we need to increment the pointer by to move to the next row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # Block size is the next power of two greater than n_cols, allowing us to
    # fit each row into a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM using a mask since BLOCK_SIZE may be larger than n_cols
    row = tl.load(input_ptrs, cache_modifier=".cg")
    # Subtract the maximum value for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note: Exponential operation in Triton is fast but approximate (i.e., imagine __expf in CUDA)
    numerator = tl.exp2(row_minus_max * LOG2E)
    denominator = tl.sum(numerator, axis=0)

    inv_denominator = 1.0 / denominator
    softmax_output = numerator * inv_denominator
    # Write the output back to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, cache_modifier=".cs")

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

def softmax(x):
    """
    Computes the row-wise softmax of a 2D input tensor.

    Key parameters:
        x (torch.Tensor): A 2D input tensor.

    Returns:
        torch.Tensor: A tensor of the same shape as 'x', where softmax has been
        applied along the last dimension (row-wise).

    Note:
        - The input tensor 'x' must reside on the GPU.
    """
    _LOGGER.info(f"SOFTMAX: x={tuple(x.shape)}")
    n_rows, n_cols = x.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    print("MAX_FUSED_SIZE: ", MAX_FUSED_SIZE)
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    print("BLOCK_SIZE: ", BLOCK_SIZE)
    y = torch.empty_like(x)

    waves_per_eu = 4 # 2
    num_warps = 4
    num_stages = 2

    num_programs = n_rows

    grid = lambda meta: (num_programs,)  # noqa: E731
    # _softmax_kernel_online[grid](
    _softmax_kernel[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return y
