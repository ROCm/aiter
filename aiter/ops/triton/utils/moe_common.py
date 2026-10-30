import torch


def torch_silu_and_mul_ref(input):
    """
    Performs the SiLU activation on the first half of the input tensor and
    multiplies it element-wise with the second half.
    Args:
        input (torch.Tensor): Input tensor of shape [..., 2 * d].
        param (float): Parameter for the SiLU activation function.
    Returns:
        torch.Tensor: Output tensor of shape [..., d].
    """
    dtype = input.dtype
    d = input.size(-1) // 2
    A, B = input[:, :d], input[:, d:]

    silu_A = A / (1.0 + torch.exp(-A.float()))

    output = silu_A * B

    return output.to(dtype)


def can_overflow_int32(tensor: torch.Tensor):
    """True when the tensor's largest linear offset does not fit in int32."""
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32
