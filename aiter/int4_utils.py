import torch

# AMD
#    packed_4_bits (pack)   = [0, 2, 4, 6, 1, 3, 5, 7]
#                  (unpack) = [0, 4, 1, 5, 2, 6, 3, 7]


#This code is adapted from https://github.com/ROCm/vllm/blob/main/vllm/model_executor/layers/quantization/awq_triton.py

#zeros are ignored since we use symmetric quantization
# qweight is both quantized and bit-packed alone the same row. All the bits in the same row has the same scaling factor.
# 8 INT4s are packed into one INT32. INT4 instead of UINT4 is used.

################################################################################
# Custom Triton Kernel & Wrapper 
################################################################################

def convert_int8_to_uint32_int4(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.dtype == torch.int8, "input should be int8"

    if tensor.shape[-1] % 8 != 0:
        raise ValueError("k % 8 should be zero")

    tensor_reshaped = tensor.reshape(*tensor.shape[:-1], tensor.shape[-1] // 8, 8)
    high_bits = ((tensor_reshaped & 0x0F))
    merged = (
        (high_bits[:, :, :, 7].to(torch.int32) << 28) |
        (high_bits[:, :, :, 6].to(torch.int32) << 24) |
        (high_bits[:, :, :, 5].to(torch.int32) << 20) |
        (high_bits[:, :, :, 4].to(torch.int32) << 16) |
        (high_bits[:, :, :, 3].to(torch.int32) << 12) |
        (high_bits[:, :, :, 2].to(torch.int32) << 8)  |
        (high_bits[:, :, :, 1].to(torch.int32) << 4)  |
         high_bits[:, :, :, 0].to(torch.int32)
    )
    return merged.view(dtype=torch.uint32)

def rearrange_4bit_elements(tensor):
    """
    GPU-optimized version for rearranging 4-bit segments within 32-bit integers
    [e0, e1, e2, e3, e4, e5, e6, e7] -> [e0, e2, e4, e6, e1, e3, e5, e7]
    """
    t_ = tensor.view(dtype=torch.int32)
 
    return (
        ((t_ & 0xF0000000) << 0) |   # e0 (bits 28-31)
        ((t_ & 0x00F00000) << 4) |   # e2 -> position 24-27
        ((t_ & 0x0000F000) << 8) |   # e4 -> position 20-23
        ((t_ & 0x000000F0) << 12) |  # e6 -> position 16-19
        ((t_ & 0x0F000000) >> 12) |  # e1 -> position 12-15
        ((t_ & 0x000F0000) >> 8) |   # e3 -> position 8-11
        ((t_ & 0x00000F00) >> 4) |   # e5 -> position 4-7
        (t_ & 0x0000000F)            # e7 (bits 0-3)
    ).view(dtype=torch.uint32)

