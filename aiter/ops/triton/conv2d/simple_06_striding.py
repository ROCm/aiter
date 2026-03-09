import torch
import torch.nn as nn
import triton
import triton.language as tl
import pandas as pd
import numpy as np
import pytest
import time


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
    
    # //==============start of threads===============//
    pid_h = tl.program_id(axis=0) 
    start_pid_h = (pid_h*stride_h) - pad_h
    kernel_h_offset = tl.arange(0, BLOCK_SIZE_H)
    h_offset = (start_pid_h + kernel_h_offset)

    pid_w = tl.program_id(axis=1)
    start_pid_w = (pid_w*stride_w) - pad_w
    kernel_w_offset = tl.arange(0, BLOCK_SIZE_W)
    w_offset = (start_pid_w + kernel_w_offset)

    pid_k = tl.program_id(axis=2)
    num_pid_k = tl.num_programs(axis=2)
    start_pid_k = pid_k

    channels = tl.arange(0, BLOCK_SIZE_C)
    # //===============end of threads===============//

    #define starting position for sliding mask
    filter_h = (start_pid_h + H_k)
    filter_w = (start_pid_w + W_k)

    #mask for input and kernel
    mask_h = (h_offset < filter_h) & (h_offset >= 0) & (h_offset < (H_i))
    mask_w = (w_offset < filter_w) & (w_offset >= 0) & (w_offset < (W_i))
    mask_c = (channels < C_i)
    mask_2d = (mask_h[None, :, None]) & (mask_w[None, None, :]) & (mask_c[:, None, None])

    
    offset_input = ((h_offset[None, :, None]*stride_input_h) + (w_offset[None,None,:]*stride_input_w) + (channels[:, None, None]*stride_input_c))
    offset_kernel = ((kernel_h_offset[None, :, None]*stride_kernel_h) + (kernel_w_offset[None,None,:]*stride_kernel_w) + (channels[:, None, None]*stride_kernel_c))

    
    for n in range(0, N_i):
        z_sum = tl.zeros((1,), dtype=tl.float32)
        x = tl.load(input_ptr + offset_input + (n*stride_input_n), mask=mask_2d, other=0.0)
        y = tl.load(kernel_ptr + offset_kernel + (start_pid_k*stride_kernel_n), mask=mask_2d, other=0.0)
        z_sum += tl.sum(x*y)

        offset_pid_h = pid_h
        offset_pid_w = pid_w
        offset_pid_k = start_pid_k
        offset_output = ((offset_pid_h[:,None]*stride_output_h) + (offset_pid_w[None,:]*stride_output_w) + (offset_pid_k*stride_output_c) + (n*stride_output_n))
        #mask for output
        mask_output_h = (offset_pid_h < H_out) 
        mask_output_w = (offset_pid_w < W_out) 
        mask_output = mask_output_h[:, None] & mask_output_w[None, :]
        
        # tl.store(output_ptr+offset_output, z)
        tl.store(output_ptr+offset_output, z_sum, mask=mask_output)

"""
conv2d()

Helper function to pass in tensors objects and setup requirements for kernel
"""
def conv2d(input_tensor: torch.Tensor , kernel_tensor: torch.Tensor, stride=(1,1), padding=(0,0), dilation=(1,1)):
    
    N_i, C_i, H_i, W_i = input_tensor.shape
    print(f'Input Shape Details: -> N_i:{N_i}, C_i:{C_i}, H_i:{H_i}, W_i:{W_i}')
    N_k, C_k, H_k, W_k = kernel_tensor.shape
    print(f'Kernel Shape Details: -> N_k:{N_k}, C_k:{C_k}, H_k:{H_k}, W_k:{W_k}')

    assert C_i == C_k, "Channels must match across input and kernel"

    pad_h, pad_w = padding
    dilat_h, dilat_w = dilation
    stride_h, stride_w = stride
    BLOCK_SIZE_W = 32
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_C = 128

    print("==================================================================")
    print(f'Stride Details: Input-> Stride 0: {input_tensor.stride(0)} Stride 1:{input_tensor.stride(1)}, Stride 2:{input_tensor.stride(2)}, Stride 3: {input_tensor.stride(3)} ')
    print(f'Stride Details: Kernel-> Stride 0: {kernel_tensor.stride(0)} Stride 1:{kernel_tensor.stride(1)}, Stride 2:{kernel_tensor.stride(2)}, Stride 3: {kernel_tensor.stride(3)} ')
    print("==================================================================")

    H_out = ((H_i+2*pad_h-dilat_h*(H_k-1)-1)//stride_h) + 1
    W_out = ((W_i+2*pad_w-dilat_w*(W_k-1)-1)//stride_w) + 1
    
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
  

def unit_test_square_no_padding_conv2d():
    N_i=1; C_i=1; H_i=32; W_i=32
    N_k=1; C_k=1; H_k=3; W_k=3
    pad_h=0; pad_w=0
    stride_h=1; stride_w=1
    input_data = torch.randn( (N_i,C_i,H_i,W_i), device=DEVICE, dtype=torch.float32)
    kernel = torch.randn( (N_k,C_k,H_k,W_k), device=DEVICE, dtype=torch.float32)
    torch_conv2d= nn.Conv2d(in_channels=C_i,out_channels=C_k,kernel_size=(H_k,W_k), stride=(stride_h, stride_w), padding=(pad_h, pad_w), bias=False, dtype=torch.float32)
    with torch.no_grad():
        torch_conv2d.weight.data = kernel
    triton_output = conv2d(input_data, kernel, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    torch_output = torch_conv2d(input_data)
    result = torch.allclose(triton_output, torch_output, atol=0.01, rtol=0.01)
    if result:
        print("✅ Square No Padding Conv2d Test Passed")
        print(f'Triton Shape: {triton_output.shape} | Torch Shape: {torch_output.shape}')
        print("==========================================END OF TEST============================================\n")
    else:
        print("❌ Square No Padding Conv2d Test Failed")
        print(f'Triton Shape: {triton_output.shape} | Torch Shape: {torch_output.shape}')
        print(f'Triton Output: {triton_output}')
        print(f'Torch Output: {torch_output}')
        print("==========================================END OF TEST============================================\n")

def unit_test_square_padding_conv2d():
    N_i=1; C_i=1; H_i=32; W_i=32
    N_k=1; C_k=1; H_k=3; W_k=3
    pad_h=1; pad_w=1
    stride_h=1; stride_w=1
    input_data = torch.randn( (N_i,C_i,H_i,W_i), device=DEVICE, dtype=torch.float32)
    kernel = torch.randn( (N_k,C_k,H_k,W_k), device=DEVICE, dtype=torch.float32)
    torch_conv2d= nn.Conv2d(in_channels=C_i,out_channels=C_k,kernel_size=(H_k,W_k), stride=(stride_h, stride_w), padding=(pad_h, pad_w), bias=False, dtype=torch.float32)
    with torch.no_grad():
        torch_conv2d.weight.data = kernel
    triton_output = conv2d(input_data, kernel, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    torch_output = torch_conv2d(input_data)
    result = torch.allclose(triton_output, torch_output, atol=0.01, rtol=0.01)
    if result:
        print("✅ Square Padding Conv2d Test Passed")
        print(f'Triton Shape: {triton_output.shape} | Torch Shape: {torch_output.shape}')
        print("==========================================END OF TEST============================================\n")
    else:
        print("❌ Square Padding Conv2d Test Failed")
        print(f'Triton Shape: {triton_output.shape} | Torch Shape: {torch_output.shape}')
        print("==========================================END OF TEST============================================\n")

def unit_test_square_stride_conv2d():
    N_i=1; C_i=1; H_i=32; W_i=33
    N_k=1; C_k=1; H_k=3; W_k=3
    pad_h=0; pad_w=0
    stride_h=2; stride_w=2
    input_data = torch.randn( (N_i,C_i,H_i,W_i), device=DEVICE, dtype=torch.float32)
    kernel = torch.randn( (N_k,C_k,H_k,W_k), device=DEVICE, dtype=torch.float32)
    torch_conv2d= nn.Conv2d(in_channels=C_i,out_channels=C_k,kernel_size=(H_k,W_k), stride=(stride_h, stride_w), padding=(pad_h, pad_w), bias=False, dtype=torch.float32)
    with torch.no_grad():
        torch_conv2d.weight.data = kernel
    triton_output = conv2d(input_data, kernel, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    torch_output = torch_conv2d(input_data)
    result = torch.allclose(triton_output, torch_output, atol=0.01, rtol=0.01)
    if result:
        print("✅ Square Stride Conv2d Test Passed")
        print(f'Triton Shape: {triton_output.shape} | Torch Shape: {torch_output.shape}')
        print("==========================================END OF TEST============================================\n")
    else:
        print("❌ Square Stride Conv2d Test Failed")
        print(f'Triton Shape: {triton_output.shape} | Torch Shape: {torch_output.shape}')
        print("==========================================END OF TEST============================================\n")

def unit_test_odd_padding_conv2d():
    N_i=9; C_i=77; H_i=224; W_i=32
    N_k=3; C_k=77; H_k=17; W_k=17
    pad_h=5; pad_w=3
    stride_h=1; stride_w=1
    input_data = torch.randn( (N_i,C_i,H_i,W_i), device=DEVICE, dtype=torch.float32)
    kernel = torch.randn( (N_k,C_k,H_k,W_k), device=DEVICE, dtype=torch.float32)
    torch_conv2d= nn.Conv2d(in_channels=C_i,out_channels=C_k,kernel_size=(H_k,W_k), stride=(stride_h, stride_w), padding=(pad_h, pad_w), bias=False, dtype=torch.float32)
    with torch.no_grad():
        torch_conv2d.weight.data = kernel
    triton_output = conv2d(input_data, kernel, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    torch_output = torch_conv2d(input_data)
    result = torch.allclose(triton_output, torch_output, atol=0.01, rtol=0.01)
    if result:
        print("✅ Odd Padding Conv2d Test Passed")
        print(f'Triton Shape: {triton_output.shape} | Torch Shape: {torch_output.shape}')
        print("==========================================END OF TEST============================================\n")
    else:
        print("❌ Odd Padding Conv2d Test Failed")
        print(f'Triton Shape: {triton_output.shape} | Torch Shape: {torch_output.shape}')
        print("==========================================END OF TEST============================================\n")

def unit_test_odd_padding_stride_conv2d():
    N_i=9; C_i=77; H_i=224; W_i=32
    N_k=3; C_k=77; H_k=17; W_k=17
    pad_h=5; pad_w=3
    stride_h=3; stride_w=2
    input_data = torch.randn( (N_i,C_i,H_i,W_i), device=DEVICE, dtype=torch.float32)
    kernel = torch.randn( (N_k,C_k,H_k,W_k), device=DEVICE, dtype=torch.float32)
    torch_conv2d= nn.Conv2d(in_channels=C_i,out_channels=C_k,kernel_size=(H_k,W_k), stride=(stride_h, stride_w), padding=(pad_h, pad_w), bias=False, dtype=torch.float32)
    with torch.no_grad():
        torch_conv2d.weight.data = kernel
    triton_output = conv2d(input_data, kernel, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    torch_output = torch_conv2d(input_data)
    result = torch.allclose(triton_output, torch_output, atol=0.01, rtol=0.01)
    if result:
        print("✅ Odd Padding Stride Conv2d Test Passed")
        print(f'Triton Shape: {triton_output.shape} | Torch Shape: {torch_output.shape}')
        print("==========================================END OF TEST============================================\n")
    else:
        print("❌ Odd Padding Stride Conv2d Test Failed")
        print(f'Triton Shape: {triton_output.shape} | Torch Shape: {torch_output.shape}')
        print("==========================================END OF TEST============================================\n")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N_i", "C_i", "N_k", "C_k"],
        x_vals = [2* i for i in range(2,100, 1)],
        line_arg='id',
        line_vals = ['triton', 'torch'],
        line_names = ["Triton", "Torch"],
        plot_name="Conv2d Benchmark",
        styles=[("green", "-"), ("blue", "-")],
        ylabel="Mean Runtime",
        args={'H_i':32, 'W_i':32, 'H_k':3, 'W_k':3, 'pad_h':1, 'pad_w':1, 'stride_h':1, 'stride_w':1, 'dilat_h':1, 'dilat_w':1},
    )
) 

def benchmark(N_i, C_i, H_i, W_i, N_k, C_k, H_k, W_k, pad_h, pad_w, stride_h, stride_w, dilat_h, dilat_w, id):
    input_data = torch.randn( (N_i,C_i,H_i,W_i), device=DEVICE, dtype=torch.float32)
    kenel_data = torch.randn( (N_k,C_k,H_k,W_k), device=DEVICE, dtype=torch.float32)
    torch_conv2d= nn.Conv2d(in_channels=C_i,out_channels=C_k,kernel_size=(H_k,W_k), stride=(stride_h, stride_w), padding=(pad_h, pad_w), bias=False, dtype=torch.float32)
    with torch.no_grad():
        torch_conv2d.weight.data = kenel_data
    if id == 'triton':
        mean_result = triton.testing.do_bench(lambda: conv2d(input_data, kenel_data, stride=(stride_h, stride_w), padding=(pad_h, pad_w)))
        mean_result = mean_result
    if id == 'torch':
        mean_result = triton.testing.do_bench(lambda: torch_conv2d(input_data))
        mean_result = mean_result
     
    return mean_result

if __name__ == "__main__":
    torch.manual_seed(0)
    print("==========================================START OF TEST==========================================\n")
    unit_test_square_no_padding_conv2d()
    print("==========================================START OF TEST==========================================\n")
    unit_test_square_padding_conv2d()
    print("==========================================START OF TEST==========================================\n")
    unit_test_square_stride_conv2d()
    print("==========================================START OF TEST==========================================\n")
    unit_test_odd_padding_conv2d()
    print("==========================================START OF TEST==========================================\n")
    unit_test_odd_padding_stride_conv2d()

    #wait 3 seconds before running the benchmark
    # time.sleep(3)
    # benchmark.run(show_plots=True)

