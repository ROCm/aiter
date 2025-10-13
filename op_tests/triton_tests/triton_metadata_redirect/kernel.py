import triton
import triton.language as tl


@triton.jit
def empty_kernel(x_ptr, SIZE: tl.constexpr):
    # Add implementation here
    return
