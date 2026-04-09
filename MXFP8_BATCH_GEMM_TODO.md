# MXFP8 Batched GEMM ASM Integration — Next Steps

## 已完成（by Agent）

- [x] `hsa/gfx1250/mxfp8_batch_gemm/mxfp8_batch_gemm.csv` — CSV 已填写（4个非 cluster kernel）
- [x] `hsa/gfx1250/mxfp8_batch_gemm/build_co.sh` — .co 编译脚本（从 .s 生成 .co）
- [x] `csrc/py_itfs_cu/asm_mxfp8_batch_gemm.cu` — C++ dispatch（KernelArgs 匹配 mxfp8fp4gemm.cpp）
- [x] `aiter/jit/optCompilerConfig.json` — 新增 `module_mxfp8_batch_gemm_asm` JIT 配置
- [x] `aiter/ops/batched_gemm_op_a8w8.py` — ctypes 桩函数 `_mxfp8_batch_gemm_asm` + 重写 `batched_gemm_a8w8_ASM()`
- [x] `op_tests/test_batched_gemm_a8w8.py` — ASM 测试函数 + `--mode asm` 参数

## .co 命名规范

参考 aiter 仓库中已有 .co 命名风格（小写+下划线+描述性后缀）：
- gfx942: `I8gemm_bf16_perTokenI8_BpreShuffle_112x256.co`, `fmoe_fp8_g1u1_multix_subGU_128.co`
- gfx950: `bwd_hd128_bf16_a16_psskddv.co`

本次 .co 命名规则: `mxfp8_batch_gemm_{tile_m}x{tile_n}[_variant].co`

| .s 文件 (shaders/) | knl_name (kernel 符号) | .co 输出文件名 | tile_m | tile_n |
|---|---|---|---|---|
| `MXFP8_GEMM_1TG_4W_64mx1_128nx4.s` | `MXFP8_GEMM_1TG_4W_64mx1_128nx4_KERNEL_FUNC` | `mxfp8_batch_gemm_64x512.co` | 64 | 512 |
| `MXFP8_GEMM_1TG_4W_64mx1_128nx4_APRESHUFFLE.s` | `MXFP8_GEMM_1TG_4W_64mx1_128nx4_APRESHUFFLE_KERNEL_FUNC` | `mxfp8_batch_gemm_64x512_apreshuffle.co` | 64 | 512 |
| `MXFP8_GEMM_1TG_4W_128mx2_128nx2.s` | `MXFP8_GEMM_1TG_4W_128mx2_128nx2_KERNEL_FUNC` | `mxfp8_batch_gemm_256x256.co` | 256 | 256 |
| `MXFP8FP4_GEMM_1TG_4W_128mx2_128nx2.s` | `MXFP8FP4_GEMM_1TG_4W_128mx2_128nx2_KERNEL_FUNC` | `mxfp8fp4_batch_gemm_256x256.co` | 256 | 256 |

Cluster 变体（需先支持 cluster launch）：

| .s 文件 (shaders/) | knl_name | .co 输出文件名 |
|---|---|---|
| `MXFP8_GEMM_1TG_4W_64mx1_128nx4_CLUSTER1x4.s` | `..._CLUSTER1x4_KERNEL_FUNC` | `mxfp8_batch_gemm_64x512_cluster1x4.co` |
| `MXFP8_GEMM_1TG_4W_64mx1_128nx4_APRESHUFFLE_CLUSTER1x4.s` | `..._APRESHUFFLE_CLUSTER1x4_KERNEL_FUNC` | `mxfp8_batch_gemm_64x512_apreshuffle_cluster1x4.co` |
| `MXFP8_GEMM_1TG_4W_128mx2_128nx2_CLUSTER4x4.s` | `..._CLUSTER4x4_KERNEL_FUNC` | `mxfp8_batch_gemm_256x256_cluster4x4.co` |
| `MXFP8FP4_GEMM_1TG_4W_128mx2_128nx2_CLUSTER4x4.s` | `..._CLUSTER4x4_KERNEL_FUNC` | `mxfp8fp4_batch_gemm_256x256_cluster4x4.co` |

## 你需要做的

### 1. 生成 .co 文件（在有 ROCm 的机器上）

已提供编译脚本 `hsa/gfx1250/mxfp8_batch_gemm/build_co.sh`。

**前置条件**：
- .s 文件已生成（在 `poc_kl/mi400/mxfp8fp4gemm/shaders/` 下）
- 有 `amdclang++` 或 `/opt/rocm/llvm/bin/clang++`

```bash
# 方法 1: 用编译脚本（推荐）
cd /local_vol1_nobackup/zw/aiter/hsa/gfx1250/mxfp8_batch_gemm
./build_co.sh

# 方法 2: 手动编译单个
amdclang++ -x assembler -target amdgcn--amdhsa --offload-arch=gfx1250 \
    /local_vol1_nobackup/zw/poc_kl/mi400/mxfp8fp4gemm/shaders/MXFP8_GEMM_1TG_4W_64mx1_128nx4.s \
    -o mxfp8_batch_gemm_64x512.co
```

如果 .s 文件还未生成，先跑 convert：
```bash
cd /local_vol1_nobackup/zw/poc_kl/mi400/mxfp8fp4gemm
./run.sh convert
```

### 2. 验证 .co 文件已到位

```bash
ls -la /local_vol1_nobackup/zw/aiter/hsa/gfx1250/mxfp8_batch_gemm/*.co
# 应该看到:
# mxfp8_batch_gemm_64x512.co
# mxfp8_batch_gemm_64x512_apreshuffle.co
# mxfp8_batch_gemm_256x256.co
# mxfp8fp4_batch_gemm_256x256.co
```

CSV 已填好（`mxfp8_batch_gemm.csv`），无需额外编辑。

### 3. 实现 preshuffle 工具函数

ASM kernel 要求输入已 preshuffle。参考 `mxfp8fp4gemm.cpp` 和 `common/fmoe.hpp` 中的 `moe_shuffle` / `moe_shuffle_one`：

- **A preshuffle** (`a_preshuffle=1`): `(m, k) → (m/2, k/128, 2, 128)`
  - 调 `moe_shuffle<uint8>(A, batch, M, K, true, 128, 2)`
- **B preshuffle** (`b_preshuffle=1`): `(n, k) → (n/16, k/16, 16, 16)`
  - 调 `moe_shuffle<uint8>(B, batch, N, K, true, TSRCB, LAYOUT_16X16)`
- **Scale preshuffle**: `(dim, k/32) → (dim/32, k/4, 32, 4)`
  - 调 `moe_shuffle_one(Scale, batch*dim, K/32, true, 4, 32)`

可在 Python 侧用 `torch.Tensor.reshape().permute().contiguous()` 实现：
```python
# A preshuffle: (B, M, K) -> (B, M/2, K/128, 2, 128) -> (B, M, K) contiguous
A = A.reshape(B, M//2, 2, K//128, 128).permute(0, 1, 3, 2, 4).contiguous().reshape(B, M, K)

# B preshuffle: (B, N, K) -> (B, N/16, 16, K/16, 16) -> (B, N/16, K/16, 16, 16) -> (B, N, K) contiguous
B = B.reshape(B, N//16, 16, K//16, 16).permute(0, 1, 3, 2, 4).contiguous().reshape(B, N, K)

# Scale preshuffle: (B, dim, K/32) -> (B, dim/32, 32, K/32/4, 4) -> (B, dim/32, K/32/4, 32, 4)
S = S.reshape(B, dim//32, 32, scale_k//4, 4).permute(0, 1, 3, 2, 4).contiguous().reshape(B, dim, scale_k)
```

### 4. 设置环境变量并测试

```bash
export AITER_GPU_ARCHS="gfx1250"
export AITER_ASM_DIR="/local_vol1_nobackup/zw/aiter/hsa"

# 测试 ASM kernel
cd /local_vol1_nobackup/zw/aiter
python op_tests/test_batched_gemm_a8w8.py -m asm -s 64,512,1024 -b 16
```

### 5. （可选）Cluster launch 支持

当前 `AiterAsmKernel::launch_kernel()` 不支持 cluster launch（`hipDrvLaunchKernelEx`）。
如果需要使用 cluster kernel 变体（如 `_CLUSTER1x4`），需要扩展 `csrc/include/aiter_hip_common.h` 中的 `AiterAsmKernel` 类或在 `asm_mxfp8_batch_gemm.cu` 中直接调 HIP API。

## 文件对照表

| 文件 | 用途 | 状态 |
|------|------|------|
| `hsa/gfx1250/mxfp8_batch_gemm/*.co` | ASM kernel 二进制 | **待编译**（在有 ROCm 的机器上跑 `build_co.sh`） |
| `hsa/gfx1250/mxfp8_batch_gemm/mxfp8_batch_gemm.csv` | kernel 元数据 | ✅ 已填写 |
| `hsa/gfx1250/mxfp8_batch_gemm/build_co.sh` | .co 编译脚本 | ✅ 已完成 |
| `csrc/py_itfs_cu/asm_mxfp8_batch_gemm.cu` | C++ dispatch | ✅ 已完成 |
| `aiter/jit/optCompilerConfig.json` | JIT 模块配置 | ✅ 已完成 |
| `aiter/ops/batched_gemm_op_a8w8.py` | Python 接口 | ✅ 已完成 |
| `op_tests/test_batched_gemm_a8w8.py` | 测试 | ✅ 已完成（preshuffle TODO） |
