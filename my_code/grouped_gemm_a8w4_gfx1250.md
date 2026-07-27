# gfx1250 A8W4 Grouped MoE GEMM Kernel 说明

本文说明 `run_gemm.sh` 对应日志中的两个 GEMM kernel：

```text
gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1
gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1
```

分析对应以下固定 workload：

- GPU：gfx1250，wave size 为 32。
- 数据格式：A8W4，即 activation 为 MXFP8 E4M3，weight 为 MXFP4 E2M1。
- 量化粒度：`per_1x32`，K 方向每连续 32 个元素共享一个 E8M0 scale。
- 输出 dtype：BF16；矩阵乘累加使用 FP32 accumulator。
- token 数 `T = 4096`。
- expert 数 `E = 384`。
- `topk = 6`。
- model/hidden dimension `H = 7168`。
- intermediate dimension `I = 768`。
- stage1 weight layout：GUGU，即 `[gate0, up0, gate1, up1, ...]`。
- activation：SiLU。
- stage1 和 stage2 都带 bias。

## 1. Route 与 contiguous-M 布局

总 route 数为：

```text
R = T * topk = 4096 * 6 = 24576
```

测试设置了 `AITER_MOE_EXPERT_BALANCE=true`。因为 `24576 / 384 = 64`，所以每个 expert 恰好接收 64 条 route。两个 GEMM 的 `tile_m` 都是 16，因此每个 expert 对应 4 个有效 M tile。

TDM grouped GEMM 不使用 `(E, max_m, K)` 的稠密 activation buffer，而是把所有 expert 的行拼到一个 contiguous-M buffer。当前参数下：

```text
align_m     = 16
valid_M     = 24576
contiguous_M
  = align_up(T * topk + E * align_m - topk, align_m)
  = align_up(24576 + 384 * 16 - 6, 16)
  = 30720
```

因此 activation/output buffer 的前导形状是 `(1, 30720, ...)`：

- 前 24576 行是有效 route。
- 后 6144 行是静态容量余量。
- 第一维 `1` 不是 expert 维；expert 已经折叠进 contiguous-M 维。

`m_tile_map` 是形状 `(384,)` 的 `torch.int32` tensor，记录每个 expert 的有效结束行。当前均衡 workload 中：

```text
m_tile_map[e] = (e + 1) * 64
```

GEMM kernel 对每个 M tile 在 `m_tile_map` 中二分查找所属 expert，再从该 expert 的 weight、weight scale 和 bias slab 取数据。落在 24576 之后的 tile 得到 sentinel expert id `384`，随后被跳过。

## 2. 数据的存储 dtype 与数学 dtype

这些 kernel 的 MX 数据使用“字节存储 + kernel 内解释”的方式：

- MXFP8 activation：
  - PyTorch storage dtype 是 `torch.uint8`。
  - 每个 byte 在 kernel 中解释为一个 FP8 E4M3 数值。
- MXFP4 weight：
  - PyTorch storage dtype 是 `torch.uint8`。
  - 每个 byte 打包两个 FP4 E2M1 数值，因此物理 K 维是逻辑 K 维的一半。
- A/W scale：
  - storage dtype 是 `torch.uint8`，每个 byte 是一个 E8M0 scale。
  - kernel launch 前会对其做 `view(torch.int32)`，以 4 个 E8M0 byte 为一个 dword 加载；这只是 bit view，不是数值转换。
- GEMM accumulator：FP32。
- bias 和 GEMM 输出：BF16。

weight 在进入 kernel 前已经完成 gfx1250 WMMA/TDM preshuffle。weight scale 也已经转换为 `n32k4` 布局。因此下面列出的 weight/scale 形状是物理 shape，不能按普通 row-major tensor 直接解释。

## 3. GEMM1：Gate/Up GEMM + SiLU

### 3.1 Kernel 名

```text
gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1
```

名称中的配置含义：

- `t16x256x256`：`tile_m=16, tile_n=256, tile_k=256`。
- `w1x4`：一个 workgroup 使用 `m_warp=1, n_warp=4`，共 4 个 wave32，即 128 个线程。
- `b2`：A、B、A-scale、B-scale 使用两级 TDM/LDS ring buffer。
- `e384`：384 个 expert。
- `afp8`：activation operand 解释为 FP8 E4M3。
- `outbf16`：输出为 BF16。
- `silu`：融合 gate/up SiLU。
- `bias1`：`has_bias=1`，不是“第一个 bias”的编号。
- `qout0`：不在 GEMM1 epilogue 内量化输出。
- `qrep1`：output quant scale 的 WMMA row replication 为 1；由于 `qout0`，本次不会使用。
- `v1`：TDM descriptor version 1。

### 3.2 Tensor/pointer 输入

#### `arg_a` / `a1_payload`

- storage dtype：`torch.uint8`。
- 数学 dtype：MXFP8 E4M3。
- 物理 shape：`(1, 30720, 7168)`。
- 逻辑 shape：`(valid_M, K) = (24576, 7168)`。
- 来源：原始 BF16 hidden `(4096, 7168)` 按 topk route 复制、scatter 到 contiguous-M 行，同时做 per-1x32 MXFP8 quant。

#### `arg_scale_a` / `a1_scale`

- storage dtype：`torch.uint8` E8M0。
- byte shape：`(1, 30720, 224)`，其中 `224 = 7168 / 32`。
- kernel ABI view：`torch.int32`，shape 为 `(1, 30720, 56)`。
- 每个有效 A 行的每 32 个 K 元素共享一个 scale。
- scale 已按 `wmma_rep=1` 做 preshuffle。

#### `arg_b` / `w1_u8`

- storage dtype：`torch.uint8`。
- 数学 dtype：MXFP4 E2M1，每 byte 两个 FP4。
- 逻辑 shape：`(E, N, K) = (384, 1536, 7168)`。
- packed row-major shape：`(E, N, K/2) = (384, 1536, 3584)`。
- preshuffle 后 PyTorch tensor 的 storage shape 仍是 `(384, 1536, 3584)`，但 byte 顺序已经改变，不能再按普通的 `[e, n, k/2]` row-major tensor 解释。
- `N = 2 * I = 1536`。
- 行顺序是 GUGU：`[gate0, up0, gate1, up1, ...]`。
- 数据已做 gfx1250 weight preshuffle，kernel 不接收普通 row-major weight。这里要区分两个连续的转换：

1. **GGUU -> GUGU 行重排**

```text
[g0, g1, ..., g767, u0, u1, ..., u767]
    ->
[g0, u0, g1, u1, ..., g767, u767]
```

2. **gfx1250 16-row packed-tile preshuffle**

`shuffle_weight(layout=(16,16))` 对当前 uint8 tensor 执行：

```text
(384, 1536, 3584)
    -> view(384, 96, 16, 112, 2, 16)
    -> permute(0, 1, 3, 4, 2, 5)
    -> (384, 96, 112, 2, 16, 16)
    -> flatten back to storage shape (384, 1536, 3584)
```

精确 byte 映射为：

```text
src[e, n0*16 + n_lane, k0*32 + half*16 + byte]
    ->
dst[e, n0, k0, half, n_lane, byte]
```

其中：

```text
n0      = 0..95    # 1536 / 16
n_lane  = 0..15
k0      = 0..111   # 3584 / 32
half    = 0..1
byte    = 0..15
```

因此每个连续 physical tile 保存 `16 N rows x 16 packed bytes`，即
`16 N rows x 32 logical FP4 K values`。kernel 将同一块线性内存解释为：

```text
(E, N/16, (K/2)*16) = (384, 96, 57344) bytes
```

而不是普通的 `(E, N, K/2)` row-major layout。

#### `arg_scale_b` / `w1s_i32`

- 原始逻辑 scale shape：`(384, 1536, 224)` 个 E8M0 byte。
- `n32k4` preshuffle 后的 byte shape：`(384, 48, 7168)`。
  - `48 = 1536 / 32`。
  - `7168 = (7168 / 32) * 32`。
- 调用点将其 flatten 并 bit-view 成 `torch.int32`：

```text
shape = (33030144,)
```

- 每个 weight 输出行的每 32 个 K 元素共享一个 E8M0 scale。

scale preshuffle 与 weight payload 的 `N16` preshuffle 目的不同。原始 scale
的每个 N row 有 `K/32 = 224` 个 E8M0 byte；一条 `K=128` WMMAScale
需要该 row 的连续 4 个 scale byte：

```text
K 0..31, 32..63, 64..95, 96..127
```

这 4 byte 正好打包为一个 dword，供一次 `ds_load_b32` 读取。源码中的
`shuffle_scale_n32k4` 执行：

```text
(E, N, K/32)
= (384, 1536, 224)
    -> view(E, N/32, 32, (K/32)/4, 4)
    -> (384, 48, 32, 56, 4)
    -> permute(0, 1, 3, 2, 4)
    -> (384, 48, 56, 32, 4)
    -> reshape(384, 48, 7168)
```

其元素映射是：

```text
dst[e, n//32, k128_group, n%32, r]
    =
src[e, n, k128_group*4 + r]
```

其中 `r=0..3` 是一个 K128 中的 4 个 K32 scale。换句话说，物理布局为：

```text
[expert][N32 super-row][K128 group][row-in-N32][4 E8M0 bytes]
```

这里使用 `N/32` 而不是 weight payload 的 `N/16`，并不意味着逻辑 N
从 1536 变成了 48：

- weight payload 的 `N16` 对应一条 WMMA 的 16-row matrix operand；
- scale 的 `N32K4` 把两个相邻的 N16 fragment 合并成一个 32-row scale
  super-row；
- 每个 row 的 4 个 K32 scale 合成一个 dword；
- `32 rows x 4 bytes = 128 bytes`，形成适合 wave32/TDM 搬运的 scale slab。

kernel 中的读取公式直接对应这个 N32K4 布局：

```text
col_rel = wnb + wn*16 + lane16

scale_dword =
    SB[
      col_rel//32,       # N32 super-row
      ksl*32,            # 当前 K128 group
      col_rel%32         # row within N32
    ]
```

最终由一次 `ds_load_b32` 取得该 N row 在当前 K128 范围内的 4 个
E8M0 scale。preshuffle 前后元素总数保持不变：

```text
1536 * 224 = 48 * 7168 = 344064 bytes / expert
```

#### `arg_m_tile_map` / `psum`

- dtype：`torch.int32`。
- shape：`(384,)`。
- 当前值：`[64, 128, 192, ..., 24576]`。
- 用于根据 contiguous-M tile 查找对应 expert，并给最后一个部分有效 tile提供行边界。

#### `arg_bias` / `bias1`

- dtype：BF16。测试最初生成 FP32 bias，进入 TDM path 时转成 BF16。
- shape：`(384, 1536)`。
- layout 同 weight，采用 GUGU 行顺序。
- bias 在 FP32 accumulator 上相加。

#### `arg_c` / `y`

- 这是输出参数。
- dtype：BF16。
- shape：`(1, 30720, 768)`。
- GEMM 的 pre-activation N 是 1536；融合 gate/up 后输出 N 减半为 768。

#### `arg_quant_scale`

- `qout0` 时该参数不参与计算。
- wrapper 使用 `arg_c`/`y` 作为 dummy tensor 传入，不会被作为 scale 读写。

### 3.3 Scalar 与编译期参数

运行时参数：

- `i32_m = 30720`。
- `i32_n = N = 1536`。
- `f32_swiglu_limit = 7.0`。

编译期 specialization：

```text
K=7168
tile_m=16, tile_n=256, tile_k=256
m_warp=1, n_warp=4
num_buffers=2
n_experts=384
a_is_fp4=0
out_is_f16=0
stage1_act=1
has_bias=1
stage1_quant_out=0
quant_wmma_rep=1
```

### 3.4 数学计算

对 contiguous-M 中属于 expert `e` 的有效 route 行 `r`：

```text
A1f[r, k]
  = fp8_e4m3_decode(A1[r, k])
  * e8m0_decode(SA1[r, floor(k / 32)])

W1f[e, n, k]
  = fp4_e2m1_decode(W1[e, n, k])
  * e8m0_decode(SW1[e, n, floor(k / 32)])

Z1[r, n]
  = sum(k=0..7167, A1f[r, k] * W1f[e, n, k])
  + bias1[e, n]
```

`Z1` 使用 FP32 累加。GUGU 布局让相邻两个输出分别是 gate 和 up：

```text
g = min(Z1[r, 2*j], 7.0)
u = clamp(Z1[r, 2*j + 1], -7.0, 7.0)

Y[r, j] = BF16(g * sigmoid(g) * u)
```

注意 gate 只做上界 clamp，没有下界 clamp；up 同时做上下界 clamp。

等价地，每个 expert 执行：

```text
(64, 7168) @ (1536, 7168)^T
    -> (64, 1536)
    -> bias + paired SiLU
    -> (64, 768)
```

route weight 不在 GEMM1 中相乘，最终由 `moe_gather_reduce` 使用。

### 3.5 Kernel 实现流程

1. 使用 contiguous-M swizzle 将 block 映射到 `(M tile, N tile)`。
2. 根据 `m_tile_map` 二分查找当前 M tile 所属 expert。
3. 通过 TDM 将以下四类数据搬到 LDS：
   - A：MXFP8 activation。
   - B：preshuffled MXFP4 weight。
   - SA：A 的 E8M0 scale。
   - SB：B 的 `n32k4` E8M0 scale。
4. 两级 LDS ring buffer 将下一 K tile 的加载与当前 K tile 的计算重叠。
5. 使用 gfx1250 `WMMAScale<16,16,128, FP4, FP8, FP32>` 完成 scaled matrix multiply。
6. 在 FP32 accumulator 上加 BF16 bias。
7. 对 GUGU 相邻列执行 clamp 和 `silu(gate) * up`。
8. 转成 BF16，先写入 LDS，再通过 TDM store 写到 `y`。

当前 launch：

```text
M tiles = 30720 / 16 = 1920
N tiles = 1536 / 256 = 6
grid    = 1920 * 6 = 11520 workgroups
block   = 4 wave32 = 128 threads
LDS     = 77 KiB/workgroup
```

## 4. GEMM1 与 GEMM2 之间的 A2 quant

因为 GEMM1 带 bias，代码中的 `_fuse_quant = (a8w4 and bias1 is None)` 为 false；kernel 名也明确是 `qout0`。因此 GEMM1 输出 BF16 `y` 后，额外执行：

```text
moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all
```

它将：

```text
y:        BF16  (1, 30720, 768)
```

转换为：

```text
a2_payload: uint8/MXFP8 E4M3  (1, 30720, 768)
a2_scale:   uint8/E8M0        (1, 30720, 24)
```

其中 `24 = 768 / 32`。这两个 tensor 是 GEMM2 的 A 和 A-scale。

## 5. GEMM2：Down Projection GEMM

### 5.1 Kernel 名

```text
gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1
```

与 GEMM1 相比：

- tile 改为 `tile_m=16, tile_n=512, tile_k=128`。
- `noact`：不执行 SiLU/SwiGLU。
- `bias1` 仍表示 `has_bias=1`。
- `qout0`：输出保持 BF16，不做输出量化。

### 5.2 Tensor/pointer 输入

#### `arg_a` / `a2_payload`

- storage dtype：`torch.uint8`。
- 数学 dtype：MXFP8 E4M3。
- 物理 shape：`(1, 30720, 768)`。
- 逻辑 shape：`(24576, 768)`。
- 来源：对 GEMM1 的 BF16 输出进行 per-1x32 MXFP8 quant。

#### `arg_scale_a` / `a2_scale`

- storage dtype：`torch.uint8` E8M0。
- byte shape：`(1, 30720, 24)`，其中 `24 = 768 / 32`。
- kernel ABI view：`torch.int32`，shape 为 `(1, 30720, 6)`。
- scale 已按 `wmma_rep=1` 做 preshuffle。

#### `arg_b` / `w2_u8`

- storage dtype：`torch.uint8`。
- 数学 dtype：MXFP4 E2M1，每 byte 两个 FP4。
- 物理 shape：`(384, 7168, 384)`。
- 逻辑 shape：`(E, N, K) = (384, 7168, 768)`。
- stage2 不包含 gate/up，因此没有 GUGU/GGUU 区别。
- 数据已做 gfx1250 weight preshuffle。

#### `arg_scale_b` / `w2s_i32`

- 原始逻辑 scale shape：`(384, 7168, 24)` 个 E8M0 byte。
- `n32k4` preshuffle 后的 byte shape：`(384, 224, 768)`。
  - `224 = 7168 / 32`。
  - `768 = (768 / 32) * 32`。
- 调用点 flatten 并 bit-view 成 `torch.int32`：

```text
shape = (16515072,)
```

#### `arg_m_tile_map` / `psum`

- dtype：`torch.int32`。
- shape：`(384,)`。
- 与 GEMM1 使用同一个 tensor：`[64, 128, ..., 24576]`。

#### `arg_bias` / `bias2`

- dtype：BF16，由测试生成的 FP32 bias 转换而来。
- shape：`(384, 7168)`。
- 在 FP32 accumulator 上相加。

#### `arg_c` / `grouped_out`

- 这是输出参数。
- dtype：BF16。
- shape：`(1, 30720, 7168)`。
- 只保证有效 route 行有定义；静态容量尾部不会被最终 gather 读取。

#### `arg_quant_scale`

- `qout0` 时不参与计算。
- wrapper 使用 `arg_c`/`grouped_out` 作为 dummy tensor。

### 5.3 Scalar 与编译期参数

运行时参数：

- `i32_m = 30720`。
- `i32_n = N = 7168`。
- `f32_swiglu_limit = 7.0`，但 `stage1_act=0`，因此不会使用。

编译期 specialization：

```text
K=768
tile_m=16, tile_n=512, tile_k=128
m_warp=1, n_warp=4
num_buffers=2
n_experts=384
a_is_fp4=0
out_is_f16=0
stage1_act=0
has_bias=1
stage1_quant_out=0
quant_wmma_rep=1
```

### 5.4 数学计算

对属于 expert `e` 的有效 route 行 `r`：

```text
A2f[r, k]
  = fp8_e4m3_decode(A2[r, k])
  * e8m0_decode(SA2[r, floor(k / 32)])

W2f[e, n, k]
  = fp4_e2m1_decode(W2[e, n, k])
  * e8m0_decode(SW2[e, n, floor(k / 32)])

grouped_out[r, n]
  = BF16(
      sum(k=0..767, A2f[r, k] * W2f[e, n, k])
      + bias2[e, n]
    )
```

等价地，每个 expert 执行：

```text
(64, 768) @ (7168, 768)^T
    -> bias
    -> (64, 7168)
```

GEMM2 不执行 activation，也不乘 topk route weight。

### 5.5 Kernel 实现流程

GEMM2 与 GEMM1 共享同一个 TDM/WMMA kernel 模板：

1. contiguous-M swizzle 和 `m_tile_map` expert 查找。
2. TDM 搬运 A、B、SA、SB 到两级 LDS ring buffer。
3. 使用 `WMMAScale<16,16,128, FP4, FP8, FP32>` 沿 K=768 累加。
4. 在 FP32 accumulator 上加 BF16 `bias2`。
5. 不执行 activation，直接转成 BF16。
6. 经 LDS 和 TDM store 写入 `grouped_out`。

当前 launch：

```text
M tiles = 30720 / 16 = 1920
N tiles = 7168 / 512 = 14
grid    = 1920 * 14 = 26880 workgroups
block   = 4 wave32 = 128 threads
LDS     = 73 KiB/workgroup
```

## 6. 两个 GEMM 在完整 MoE 中的位置

整体数据流为：

```text
hidden BF16 (4096, 7168)
  -> route + per-1x32 MXFP8 quant + contiguous-M scatter
  -> GEMM1: A8W4 + bias1 + SiLU(gate) * up
  -> BF16 y (1, 30720, 768)
  -> per-1x32 MXFP8 quant
  -> GEMM2: A8W4 + bias2
  -> BF16 grouped_out (1, 30720, 7168)
  -> gather + topk weight + reduce
  -> final MoE output BF16 (4096, 7168)
```

因此：

- GEMM1 负责 expert 的 gate/up projection 和融合 activation。
- GEMM2 负责 expert 的 down projection。
- topk route weight 和 6 路 expert reduce 均不在这两个 GEMM 内，而是在后续 `moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16` 中完成。

## 7. 对应源码

- workload 与 tensor 构造：`op_tests/test_flydsl_grouped_gemm_gfx1250.py`
- TDM grouped MoE 调度：`aiter/ops/flydsl/grouped_moe_gfx1250.py`
- GEMM wrapper：`aiter/ops/flydsl/batched_gemm_mxfp4.py`
- GEMM kernel：`aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`
- SiLU/SwiGLU 标量 epilogue：`aiter/ops/flydsl/kernels/gemm_common_gfx1250.py`
- weight/scale preshuffle：`aiter/ops/shuffle.py`
- 当前 tuned config：`aiter/configs/tuned_grouped_fmoe.csv`
