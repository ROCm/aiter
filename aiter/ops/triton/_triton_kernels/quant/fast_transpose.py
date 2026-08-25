import triton
import triton.language as tl


@triton.jit
def _transpose_2d_kernel(
    IN_ptr,
    OUT_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_n,
    stride_out_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tiled 2D matrix transpose.

    Reads tiles from (M, N) input and writes transposed tiles to (N, M) output.
    Works with any element dtype including FP8 (e4m3, e5m2, fnuz variants).
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    in_ptrs = IN_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
    vals = tl.load(in_ptrs, mask=mask)

    out_ptrs = OUT_ptr + offs_n[None, :] * stride_out_n + offs_m[:, None] * stride_out_m
    tl.store(out_ptrs, vals, mask=mask)
