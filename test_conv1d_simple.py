#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试 AIter Causal Conv1D
"""

import torch
import torch.nn.functional as F
import aiter
import time

def torch_causal_conv1d_ref(x, weight, bias, use_silu):
    """
    PyTorch CPU 参考实现 - 使用 F.conv1d
    x: [batch, dim, seqlen]
    weight: [dim, width]
    bias: [dim] or None
    """
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    
    # Causal padding: 需要 width-1 个左侧填充
    padding_left = width - 1
    x_padded = F.pad(x, (padding_left, 0), 'constant', 0)
    
    # 调整 weight 形状用于 depthwise conv1d
    # F.conv1d 需要 weight: [out_channels, in_channels/groups, kernel_size]
    # 对于 depthwise: [dim, 1, width]
    weight_reshaped = weight.unsqueeze(1)  # [dim, 1, width]
    
    # Depthwise convolution
    out = F.conv1d(x_padded, weight_reshaped, bias=bias, groups=dim)
    
    if use_silu:
        out = out / (1 + torch.exp(-out))
    
    return out

print("=" * 80)
print("AIter Causal Conv1D 简单测试")
print("=" * 80)

# 测试配置
batch = 2
dim = 256
seqlen = 1024
width = 4
dtype = torch.float16

print(f"\n📊 测试配置:")
print(f"   Batch: {batch}")
print(f"   Dim: {dim}")
print(f"   Seqlen: {seqlen}")
print(f"   Width: {width}")
print(f"   Dtype: {dtype}")

# 创建输入张量
print("\n🔧 创建输入张量...")
x = torch.randn(batch, dim, seqlen, dtype=dtype, device="cuda")
weight = torch.randn(dim, width, dtype=dtype, device="cuda")
bias = torch.randn(dim, dtype=dtype, device="cuda")
out = torch.empty_like(x)

print(f"   x shape: {x.shape}, dtype: {x.dtype}")
print(f"   weight shape: {weight.shape}, dtype: {weight.dtype}")
print(f"   bias shape: {bias.shape}, dtype: {bias.dtype}")

# 测试 1: 基础调用（无激活）+ 准确率验证
print("\n" + "=" * 80)
print("测试 1: 基础 Causal Conv1D (无激活) + 准确率验证")
print("=" * 80)
try:
    # 运行 AIter 实现
    aiter.causal_conv1d_fwd(out, x, weight, bias, use_silu=False)
    
    # 计算 CPU 参考结果
    print("   计算 CPU 参考结果...")
    x_cpu = x.cpu().float()
    weight_cpu = weight.cpu().float()
    bias_cpu = bias.cpu().float()
    ref_cpu = torch_causal_conv1d_ref(x_cpu, weight_cpu, bias_cpu, use_silu=False)
    ref_gpu = ref_cpu.to(dtype).cuda()
    
    # 计算误差
    max_error = (out - ref_gpu).abs().max().item()
    mean_error = (out - ref_gpu).abs().mean().item()
    
    print("✅ 调用成功！")
    print(f"   out shape: {out.shape}")
    print(f"   out min: {out.min().item():.4f}, max: {out.max().item():.4f}, mean: {out.mean().item():.4f}")
    print(f"   📊 准确率:")
    print(f"      最大误差: {max_error:.2e}")
    print(f"      平均误差: {mean_error:.2e}")
    
    if max_error < 1e-2:  # fp16 精度阈值
        print(f"      ✅ 准确率验证通过！")
    else:
        print(f"      ⚠️  误差较大，可能有问题")
        
except Exception as e:
    print(f"❌ 调用失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试 2: 带 SiLU 激活 + 准确率验证
print("\n" + "=" * 80)
print("测试 2: Causal Conv1D + SiLU 激活 + 准确率验证")
print("=" * 80)
out_silu = torch.empty_like(x)
try:
    # 运行 AIter 实现
    aiter.causal_conv1d_fwd(out_silu, x, weight, bias, use_silu=True)
    
    # 计算 CPU 参考结果
    print("   计算 CPU 参考结果...")
    ref_silu_cpu = torch_causal_conv1d_ref(x_cpu, weight_cpu, bias_cpu, use_silu=True)
    ref_silu_gpu = ref_silu_cpu.to(dtype).cuda()
    
    # 计算误差
    max_error_silu = (out_silu - ref_silu_gpu).abs().max().item()
    mean_error_silu = (out_silu - ref_silu_gpu).abs().mean().item()
    
    print("✅ SiLU 测试成功！")
    print(f"   out_silu min: {out_silu.min().item():.4f}, max: {out_silu.max().item():.4f}")
    print(f"   📊 准确率:")
    print(f"      最大误差: {max_error_silu:.2e}")
    print(f"      平均误差: {mean_error_silu:.2e}")
    
    # 验证 SiLU 的效果（输出应该不同）
    diff = (out - out_silu).abs().max().item()
    print(f"   与无激活版本的最大差异: {diff:.4f}")
    if diff > 0.01:
        print("   ✅ SiLU 激活生效")
    
    if max_error_silu < 1e-2:
        print(f"   ✅ SiLU 准确率验证通过！")
    else:
        print(f"   ⚠️  SiLU 误差较大")
        
except Exception as e:
    print(f"❌ SiLU 测试失败: {e}")

# 测试 3: 无 bias + 准确率验证
print("\n" + "=" * 80)
print("测试 3: Causal Conv1D (无 bias) + 准确率验证")
print("=" * 80)
bias_empty = torch.empty(0, dtype=dtype, device="cuda")
out_no_bias = torch.empty_like(x)
try:
    # 运行 AIter 实现
    aiter.causal_conv1d_fwd(out_no_bias, x, weight, bias_empty, use_silu=False)
    
    # 计算 CPU 参考结果
    print("   计算 CPU 参考结果...")
    ref_no_bias_cpu = torch_causal_conv1d_ref(x_cpu, weight_cpu, None, use_silu=False)
    ref_no_bias_gpu = ref_no_bias_cpu.to(dtype).cuda()
    
    # 计算误差
    max_error_no_bias = (out_no_bias - ref_no_bias_gpu).abs().max().item()
    mean_error_no_bias = (out_no_bias - ref_no_bias_gpu).abs().mean().item()
    
    print("✅ 无 bias 测试成功！")
    print(f"   out_no_bias mean: {out_no_bias.mean().item():.4f}")
    print(f"   📊 准确率:")
    print(f"      最大误差: {max_error_no_bias:.2e}")
    print(f"      平均误差: {mean_error_no_bias:.2e}")
    
    if max_error_no_bias < 1e-2:
        print(f"      ✅ 无 bias 准确率验证通过！")
    else:
        print(f"      ⚠️  误差较大")
        
except Exception as e:
    print(f"❌ 无 bias 测试失败: {e}")

# 测试 4: 不同数据类型 + 准确率验证
print("\n" + "=" * 80)
print("测试 4: 不同数据类型 + 准确率验证")
print("=" * 80)

for test_dtype in [torch.float16, torch.bfloat16, torch.float32]:
    dtype_name = str(test_dtype).split('.')[-1]
    try:
        x_test = torch.randn(2, 128, 512, dtype=test_dtype, device="cuda")
        weight_test = torch.randn(128, 4, dtype=test_dtype, device="cuda")
        bias_test = torch.randn(128, dtype=test_dtype, device="cuda")
        out_test = torch.empty_like(x_test)
        
        # 运行 AIter 实现
        aiter.causal_conv1d_fwd(out_test, x_test, weight_test, bias_test, use_silu=False)
        
        # 计算 CPU 参考结果
        x_test_cpu = x_test.cpu().float()
        weight_test_cpu = weight_test.cpu().float()
        bias_test_cpu = bias_test.cpu().float()
        ref_test_cpu = torch_causal_conv1d_ref(x_test_cpu, weight_test_cpu, bias_test_cpu, use_silu=False)
        ref_test_gpu = ref_test_cpu.to(test_dtype).cuda()
        
        # 计算误差
        max_error_test = (out_test - ref_test_gpu).abs().max().item()
        
        # 根据数据类型设置不同的阈值
        threshold = 1e-2 if test_dtype in [torch.float16, torch.bfloat16] else 1e-4
        
        if max_error_test < threshold:
            print(f"   ✅ {dtype_name}: 成功 (最大误差: {max_error_test:.2e})")
        else:
            print(f"   ⚠️  {dtype_name}: 成功但误差较大 (最大误差: {max_error_test:.2e})")
            
    except Exception as e:
        print(f"   ❌ {dtype_name}: 失败 - {e}")

# 测试 5: 性能测试
print("\n" + "=" * 80)
print("测试 5: 性能测试")
print("=" * 80)

# Warmup
print("   预热中...")
for _ in range(10):
    aiter.causal_conv1d_fwd(out, x, weight, bias, use_silu=True)

# Benchmark
print("   性能测试中...")
torch.cuda.synchronize()
start = time.time()
num_iters = 100
for _ in range(num_iters):
    aiter.causal_conv1d_fwd(out, x, weight, bias, use_silu=True)
torch.cuda.synchronize()
elapsed = time.time() - start

avg_time_ms = elapsed * 1000 / num_iters
total_elements = batch * dim * seqlen
throughput = total_elements * num_iters / elapsed / 1e9

# 计算带宽
bytes_read = x.nbytes + weight.nbytes + bias.nbytes
bytes_write = out.nbytes
total_bytes = bytes_read + bytes_write
bandwidth_gb_s = total_bytes * num_iters / elapsed / 1e9

print(f"   ✅ 平均时间: {avg_time_ms:.3f} ms")
print(f"   ✅ 吞吐量: {throughput:.2f} G elements/s")
print(f"   ✅ 带宽: {bandwidth_gb_s:.2f} GB/s")

# 测试 6: 验证因果性
print("\n" + "=" * 80)
print("测试 6: 验证因果性（输出不依赖未来输入）")
print("=" * 80)

# 创建两个输入，只在未来位置不同
x1 = torch.randn(1, 4, 10, dtype=torch.float32, device="cuda")
x2 = x1.clone()
x2[:, :, -1] = 999.0  # 修改最后一个位置（未来）

out1 = torch.empty_like(x1)
out2 = torch.empty_like(x2)

weight_test = torch.randn(4, 4, dtype=torch.float32, device="cuda")
bias_test = torch.randn(4, dtype=torch.float32, device="cuda")

aiter.causal_conv1d_fwd(out1, x1, weight_test, bias_test, use_silu=False)
aiter.causal_conv1d_fwd(out2, x2, weight_test, bias_test, use_silu=False)

# 检查前 9 个位置是否相同（不应该受未来影响）
diff_past = (out1[:, :, :-1] - out2[:, :, :-1]).abs().max().item()
print(f"   前 9 个位置的最大差异: {diff_past:.6f}")
if diff_past < 1e-5:
    print("   ✅ 因果性验证通过！输出不依赖未来输入")
else:
    print("   ⚠️  因果性可能有问题")

# 最后一个位置应该不同（受当前输入影响）
diff_current = (out1[:, :, -1] - out2[:, :, -1]).abs().max().item()
print(f"   最后位置的最大差异: {diff_current:.6f}")
if diff_current > 0.1:
    print("   ✅ 当前位置正确响应输入变化")

# 总结
print("\n" + "=" * 80)
print("✅ 所有测试完成！")
print("=" * 80)
print("\n📖 更多测试:")
print("   python op_tests/test_causal_conv1d.py")
print("\n📚 查看文档:")
print("   cat csrc/kernels/CAUSAL_CONV1D_INTEGRATION.md")

