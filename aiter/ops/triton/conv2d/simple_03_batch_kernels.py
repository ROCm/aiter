import torch
import torch.nn as nn
import triton
import triton.language as tl
import pandas as pd
import numpy as np


DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def conv2d_kernel(
    # pointers to matrix
    input_ptr, kernel_ptr, output_ptr, 
    #required inputs
    N_i, C_i, H_i, W_i, N_k, C_k, H_k, W_k, pad_h, pad_w, stride_h, stride_w, dilat_h, dilat_w, H_out, W_out,
    #stride to advance to next batch, channel, row, column, 
    stride_input_n, stride_input_c, stride_input_h, stride_input_w,
    stride_kernel_n, stride_kernel_c, stride_kernel_h, stride_kernel_w,
    stride_output_n, stride_output_c, stride_output_h, stride_output_w,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    # ==== NHWC Data Format ===================== #
    # offset_nhwc(n, c, h, w) = n * HWC + h * WC + w * C + c
    # H W C D (H=height dim, W=width dim, C=Channels dim, D=Depth)
    # h w c (index of H, W, C)
    # 
    # b[0] -> Inner Dimension (C)
    # b[1] -> Width (W)
    # b[2] -> Height (H)
    # b[3] -> Batch (N)
    # ==== NHWC Data Format ===================== #
    

    # //==============start of threads===============//
    pid_h = tl.program_id(axis=0) 
    num_pid_h = tl.num_programs(axis=0)
    start_pid_h = pid_h
    kernel_h_offset = tl.arange(0, BLOCK_SIZE_H)
    h_offset = (start_pid_h + kernel_h_offset)
    # h_offset = start_pid_h

    pid_w = tl.program_id(axis=1)
    # num_pid_w = tl.num_programs(axis=1)
    start_pid_w = pid_w
    kernel_w_offset = tl.arange(0, BLOCK_SIZE_W)
    w_offset = (start_pid_w + kernel_w_offset)
    # w_offset = start_pid_w

    pid_k = tl.program_id(axis=2)
    num_pid_k = tl.num_programs(axis=2)
    start_pid_k = pid_k
   
    channels = tl.arange(0, BLOCK_SIZE_C)


    # //===============end of threads===============//

    #define starting position for sliding mask
    filter_h = start_pid_h + H_k
    filter_w = start_pid_w + W_k

    #mask for input and kernel
    mask_h = (h_offset < filter_h)
    mask_w = (w_offset < filter_w)
    mask_c = (channels < C_i)
    mask_2d = (mask_h[None, :, None]) & (mask_w[None, None, :]) & (mask_c[:, None, None])

    
    offset_input = ((h_offset[None, :, None]*stride_input_h) + (w_offset[None,None,:]*stride_input_w) + (channels[:, None, None]*stride_input_c))
    offset_kernel = ((kernel_h_offset[None, :, None]*stride_kernel_h) + (kernel_w_offset[None,None,:]*stride_kernel_w) + (channels[:, None, None]*stride_kernel_c))

    
    z_sum = tl.zeros((1,), dtype=tl.float32)

    x = tl.load(input_ptr + offset_input, mask=mask_2d, other=0.0)
    y = tl.load(kernel_ptr + offset_kernel + (start_pid_k*stride_kernel_n), mask=mask_2d, other=0.0)
    z_sum += tl.sum(x*y)
    
    #output ptr
    offset_pid_h = start_pid_h
    offset_pid_w = start_pid_w
    offset_pid_k = start_pid_k
    offset_output = ((offset_pid_h[:,None]*stride_output_h) + (offset_pid_w[None,:]*stride_output_w) + (offset_pid_k*stride_output_c))
    #mask for output
    mask_output_h = (offset_pid_h < H_out)
    mask_output_w = (offset_pid_w < W_out)
    mask_output = mask_output_h[:, None] & mask_output_w[None, :]
    
    # tl.store(output_ptr+offset_output, z)
    tl.store(output_ptr+offset_output, z_sum, mask=mask_output)


def conv2d(input_tensor: torch.Tensor , kernel_tensor: torch.Tensor, stride=(1,1), padding=(0,0), dilation=(1,1)):
    
    N_i, C_i, H_i, W_i = input_tensor.shape
    print(f'Input Shape Details: -> N_i:{N_i}, C_i:{C_i}, H_i:{H_i}, W_i:{W_i}')
    N_k, C_k, H_k, W_k = kernel_tensor.shape
    print(f'Kernel Shape Details: -> N_k:{N_k}, C_k:{C_k}, H_k:{H_k}, W_k:{W_k}')
    pad_h, pad_w = padding
    dilat_h, dilat_w = dilation
    stride_h, stride_w = stride
    BLOCK_SIZE_W = 32
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_C = 256
    print("==================================================================")
    print(f'Stride Details: Input-> Stride 0: {input_tensor.stride(0)} Stride 1:{input_tensor.stride(1)}, Stride 2:{input_tensor.stride(2)}, Stride 3: {input_tensor.stride(3)} ')
    print(f'Stride Details: Kernel-> Stride 0: {kernel_tensor.stride(0)} Stride 1:{kernel_tensor.stride(1)}, Stride 2:{kernel_tensor.stride(2)}, Stride 3: {kernel_tensor.stride(3)} ')
    print("==================================================================")

    H_out = (H_i+2*pad_h-dilat_h*(H_k-1)-1)//stride_h + 1
    W_out = (W_i+2*pad_w-dilat_w*(W_k-1)-1)//stride_w + 1
    
    print(f'H_out:{H_out}, W_out:{W_out}')
    print(f'Grid Size: {H_out}, {W_out}, {N_k}')

    output_tensor = torch.zeros( (N_i, N_k, H_out, W_out), out=None, device=DEVICE, dtype=torch.float32)

    print(f'Output Tensor Shape:{output_tensor.shape}')
    print(f'Output Tensor Stride:{output_tensor.stride()}')


    #pass on parameters and values to kernel
    grid = (H_out, W_out, N_k)
    conv2d_kernel[grid](
        #pointers
        input_tensor, kernel_tensor, output_tensor,
        #other
        N_i, C_i, H_i, W_i, N_k, C_k, H_k, W_k, pad_h, pad_w, stride_h, stride_w, dilat_h, dilat_w, H_out, W_out,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        kernel_tensor.stride(0), kernel_tensor.stride(1), kernel_tensor.stride(2), kernel_tensor.stride(3),
        output_tensor.stride(0), output_tensor.stride(1), output_tensor.stride(2), output_tensor.stride(3),
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
    )
    
    return output_tensor

if __name__ == "__main__":
    
    # batch, channel, row, column for input
    N_i=1; C_i=30; H_i=32; W_i=32
    # batch, channel, row, column for kernel filter
    N_k=10; C_k=30; H_k=5; W_k=5

    
    #Set manual seed for torch to retain similar numbers
    torch.manual_seed(0)

    #debugging purpose ignore afterwards
    col_values_input = torch.arange(1, W_i + 1,device=DEVICE, dtype=torch.float32).view(1,-1) 
    col_values_kernel = torch.arange(1, W_k + 1,device=DEVICE, dtype=torch.float32).view(1,-1)

    input_data = torch.ones( (N_i,C_i,H_i,W_i), device=DEVICE, dtype=torch.float32) *col_values_input
    #Create a random rectangle matrix
    kernel = torch.ones( (N_k,C_k,H_k,W_k), device=DEVICE, dtype=torch.float32)

    print(f'Input Tesnor: {input_data}')
    print(f'Kernel Tenosr: {kernel}')

    

    triton_out2 = conv2d(input_data, kernel) # no padding

    i_out, c_out, h_out, w_out =triton_out2.shape

    print(f'Triton Output: {triton_out2}')


    torch_conv2d= nn.Conv2d(in_channels=C_i,out_channels=C_k,kernel_size=(H_k,W_k), bias=False, dtype=torch.float32)
    with torch.no_grad():
        torch_conv2d.weight.data = kernel
   
    output_torch = torch_conv2d(input_data)
    print(f'Torch Output: {output_torch}')
    print(f'Torch Output Shape: {output_torch.shape}')
    

    if torch.allclose(triton_out2, output_torch, atol=0.01, rtol=0.01):
        print("✅ Triton and Torch match")
        # print(triton_out2.dtype)
        # print(output_torch.dtype)
    else:
        print("❌ Triton and Torch differ")
   
    