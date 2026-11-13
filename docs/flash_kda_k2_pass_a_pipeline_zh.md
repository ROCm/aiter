# FlashKDA K2 pass A（gfx950）：把 MIR 手改落到 Gluon 源码

本文讲 FlashKDA 分段 K2 递推的 pass A，硬件是 gfx950（MI350 系列）。pass A 对每个 (segment, head)，沿这个 segment 的各个 chunk 跑两条递推：h_b 从 0 开始，用真实的 v；h_a 从单位阵开始，v = 0。

研究版的 pass A kernel 做过一轮 MIR 手改：在 machine scheduler 之前改指令顺序、删多余的指令，一共 10 步。同一时段、同一张卡上，10 步全做完的版本（下文叫 MIR b1n2s4t1）比正常编译快 4.3～4.7 µs（T = 16384），每 chunk 的 ATT 周期从约 2150 降到 1663。MIR 手改进不了生产代码：替换用的 MIR 文件按 LLIR 的 sha256 匹配，源码或编译器一变就失效。

本文以上游 Triton 最新的 main 为准，逐步看这 10 步哪些能写进 Gluon 源码、怎么写、效果如何；写不出来的卡在哪里、要编译器补什么。第 3 节介绍已经放进 aiter 的 kernel：`aiter/ops/triton/_gluon_kernels/gfx950/kimi_delta_attn/flash_kda_k2.py` 里的 `k2_ab_fused_pipe_gluon`。

本文是 `flash_kda_k2_pass_a_pipeline.md` 的中文版。

**版本：**

- aiter 的 kernel 在上游 Triton main 0df169311f（2026-11-12）和 Triton 3.8.0（`3.8.0+amd.rocm7.2.0.git7cb7b059`，aiter CI 用的版本）上测过。在 Triton 3.8.0 上，aiter 不用新 kernel（3.1 节）。
- 第 2、4～6 节的研究数字是在上游 main 的 f8cddb947b（2026-11-04，`3.9.0+gitf8cddb94`）上测的；7.1 节和第 2 节的复测在 173580104f（2026-11-08）上做。两者之间 Gluon 的 AMD 部分没有提交，Triton 主体用的 LLVM（18f9e623）和 AMD 后端的 codegen 库（08b05a33）也都相同，各研究模式在两个版本上编出来的 ISA 逐字节相同。0df169311f 用的 codegen 库也是 08b05a33。
  - [triton-lang/triton#12656](https://github.com/triton-lang/triton/pull/12656) 只改了从源码构建 codegen LLVM 用的 pin（`cmake/amd-llvm-build-info.json`，293a85c0）。构建时下载哪个预编译库由 `cmake/amd-llvm-info.json` 决定，仍是 08b05a33。293a85c0 的 `SIFoldOperands` 也还不处理 `V_ADD_U32 0, x`，换过去以后第 4 步的结论不变。
- MIR 手改是在 ROCm Triton 3.7 上做的。同一个 kernel 有一个 Triton 3.7 的源码变体，用 inline asm 实现了其中两项（4.4、5.3 节）。

## 总结

### 10 步 MIR 手改在 Gluon 源码里的写法

每一步都在前面几步的基础上再加一项（1.2 节）。结果是 T = 16384、H = 12 时 kernel 时间的变化，没有注明的都是 GPU 3。"相对 `g_lt`"表示在 `g_lt` 的开关之外再开这个开关。完整数据见第 2 节。

| MIR 改动 | 源码写法 | 结果（GPU 3） | 结论 |
| --- | --- | --- | --- |
| 1. dot1：VALU 填进 MFMA 的阴影（cyc03） | `D1=3`：一条 MFMA 一段，按 MIR 的顺序；`D1=1/2`：按 piece 排 | `D1=3`：相对 `g_lt` 慢 0.65 µs；`D1=1/2`：慢 1.12 / 0.69 µs（GPU 2） | 能写，更慢 |
| 2. mem2：地址 add 提前（cyc03h） | 源码里没有对应的 op：这些 add 是 LDS 读在 lowering 时生成的。`M2=1/2` 只能把读整个挪走 | `M2=1/2`：慢 1.28 / 0.60 µs（GPU 2） | 写不出来 |
| 3a. mem1：先发 LDS 读，再发 copy（cyc03hm1spread 的前一半） | `M1=3`：两个 `sched_barrier` 把 mem1 分成 kr 读、gt 读、copy 三段 | 快 0.7 µs（GPU 2、GPU 3） | **落地，已进 aiter** |
| 3b. mem1：copy 分散到读之间（cyc03hm1spread 的后一半） | `M1=4`：kr 分 4 片读，每片后面跟一组 copy | 相对 `g_lt` 慢 0.95 µs（GPU 2） | 能写，没有收益 |
| 4. 删 `+0` 拷贝（s_z） | 没有写法。`llvm_cl_opts="disable-cgp"` 能去掉，但会 spill | 195 µs（spill） | 写不出来。要 Triton 的 AMD codegen LLVM 包含 [llvm/llvm-project#163298](https://github.com/llvm/llvm-project/pull/163298) |
| 5. 回边 SALU 挪到最后一个 `s_barrier` 之前（s_zl） | `LT=1`：copy 的基址写成循环携带的值，在 mem2 p3 推进 | 快 1.1 µs；回边 12 → 4 条 SALU | **落地，已进 aiter** |
| 6. dot2：pk 全部提前，再连发 MFMA（d2s0） | `D2=1`：在 `g * gt`、B 的 MFMA、A 的 MFMA + cvt 对之间放 `sched_barrier` | 和 MIR 同序；单开快 0.55 µs；VGPR 221 → 229 | **落地，已进 aiter** |
| 7. gt 的读提前（gt3k2） | `GTE=1/2`：mem2 先读下一个 chunk 的前 1～2 个 gt piece | −0.4～+0.9 µs，不同批次方向相反（GPU 2） | 能写，没有收益 |
| 8. β clamp 挪进 dot1（b1n2） | `BD1=2`：dot1 算 β copy 的偏移 | ATT 每 chunk 2060 → 2008；时间在噪声内 | 研究 kernel 里落地了。aiter 不需要：aiter 的 kernel 没有 β clamp |
| 9. dot2：B 的 MFMA 阴影里补发 A 的乘法（b1n2s3、b1n2s4） | `D2=4`、`D2N=3/4`：一条 MFMA 一段，指令串和 MIR 一字不差 | 相对 `g_lt` 慢 0.36 / 1.05 µs | 能写，更慢 |
| 10. MFMA 结果写回 srcC 的寄存器（tie，b1n2s4t1） | 没有写法：16x16x32 的 MFMA 没有 tie 的形式，`cd_regclass` 只能选寄存器类 | — | 写不出来。要 MFMA 提供这个选项 |

### 编译选项和研究阶段的功能

| 项目 | 研究 kernel | aiter kernel | 结果 |
| --- | --- | --- | --- |
| 关掉 SILoadStoreOptimizer | 研究用的编译器补丁 | 发射时设 `llvm_fn_attrs="target-features=-load-store-opt"` | T = 16384、H = 12：56.27 → 44.58 µs；T = 4096、H = 12：21.79 → 18.71 µs。**落地** |
| 只用 VGPR、不用 AGPR | 研究用的编译器补丁（`amdgpu-agpr-alloc=0,0`） | 不用做什么 | 不需要：上游 Triton 本来就不给这个 kernel 分 AGPR |
| kd、inv 的 B128 读 | K1 按读的列顺序写 kd、inv，一个 lane 用一条 `ds_read_b128` 读完自己的 16 字节 | 不用：要改 K1，K1 在另一个文件里 | 测不出损失：46.42 µs（graph replay），研究的 `g_lt` 是 47.06 µs |
| layout | kernel 参数 | 模块级常量 | 每次发射的主机耗时 27.5 → 13.5 µs。**落地** |
| Gluon 接口 | `buffer_load_to_shared`、`load_shared_relaxed`、`buffer_store`（Triton main 上已废弃） | buffer descriptor 接口和 `smem.load` | 时间相同。**落地** |

### aiter 里的 kernel

`k2_ab_fused_pipe_gluon` 包含落地的第 3a、5、6 步，编译时关掉 SILoadStoreOptimizer。在 gfx950 上，只要 Triton 有 `gl.amd.hint.sched_barrier` 和 buffer descriptor 接口（Triton main 2026-10-25 以后），pass A 就用它；否则 pass A 仍是 `k2_ab_fused_gluon`，行为不变。

只测 pass A，MI350X（GPU 3），Triton main 0df169311f，对比 `k2_ab_fused_gluon`，GPU 时间用 CUDA graph replay 测：

| T，H（B = 1） | segment 数 × chunk 数 | 新（µs） | 旧（µs） | 加速 |
| --- | --- | --- | --- | --- |
| 2048，12 | 16 × 4 | 14.13 | 18.09 | 1.28× |
| 4096，12 | 16 × 8 | 18.49 | 25.03 | 1.35× |
| 8192，12 | 16 × 16 | 27.65 | 40.04 | 1.45× |
| 16384，12 | 16 × 32 | 46.42 | 73.64 | 1.59× |
| 65536，4 | 32 × 64 | 74.07 | 124.25 | 1.68× |

- eager 发射快 1.30～1.89 倍；每次发射的主机耗时约 13.5 µs，旧的约 25.5 µs（3.3 节）。
- 239 或 240 个 VGPR，没有 AGPR，没有 VGPR spill。
- 落地的几步达到了研究 `g_lt` 的时间：T = 16384、H = 12 时 46.42 µs，`g_lt` 是 47.06 µs。

## 目录

1. [背景](#1-背景)
2. [10 步逐项对照](#2-10-步逐项对照)
3. [aiter 里的 kernel](#3-aiter-里的-kernel)
4. [dot2 的顺序（第 6 步）](#4-dot2-的顺序第-6-步)
5. [mem1 先读后 copy（第 3 步）](#5-mem1-先读后-copy第-3-步)
6. [回边 SALU 和 β clamp（第 5、8 步）](#6-回边-salu-和-β-clamp第-58-步)
7. [没有落地的几步（第 1、2、4、9、10 步）](#7-没有落地的几步第-124910-步)
8. [参考](#8-参考)

## 1. 背景

### 1.1 主循环

研究 kernel 是 `k2_ab_fused_wp2_pipe4u`，开头的说明：

```text
Per chunk j the loop body is four ``warp_pipeline_stage`` clusters:

  dot1[j]  DOT1 (kd @ -h, B chain seeded with v) and INV (inv @ u)
  mem1[j]  LDS read kr/gt[j]; big -> bf16; async copy kd/inv/v/beta[j+3]
  dot2[j]  h * gt and UPD (kr^T @ big); -h_B -> bf16 for chunk j+1
  mem2[j]  LDS read kd/inv/v/beta[j+1]; async copy kr/gt[j+3]
```

- 本文的研究数字都来自 mode `full_u_b128_tr`。它把主循环展开 4 次，一次迭代处理 4 个 chunk，下文叫 p0～p3。
- cluster 之间隔着 `s_barrier`，一次迭代因此分成 17 个区域：0～3 是 p0 的 dot1、mem1、dot2、mem2，4～7 是 p1，8～11 是 p2，12～15 是 p3，16 是回边（指针递增、循环计数和跳转）。p1～p3 的代码相同，p0 有自己的额外开销。
- 一个 workgroup 有 8 个 wave，分成两组（sl0、sl1），每个 SIMD 上每组各一个。两组错开一个 cluster 运行：一组做 dot cluster 时，另一组做 mem cluster，两者共用这个 SIMD 的发射。
- aiter 的 kernel 也是这 4 个 stage，同样展开 4 次（3.1 节）。

### 1.2 MIR 手改的 10 步

做法是把 machine scheduler 之前的 MIR dump 出来（`TRITON_DUMP_MIR`），用脚本改顺序或删指令，再送回编译（`TRITON_SWAP_MIR`）。每一步都在上一步的基础上再加一项：

| 步 | 名字 | 改了什么 | 累计（µs） | ATT 每 chunk |
| --- | --- | --- | --- | --- |
| — | base | 正常编译 | 0 | 2152 |
| 1 | cyc03 | dot1 按发射模型重排，VALU 填进 MFMA 的阴影；dot1 的 pk 拆成 `v_mul_f32`；β 取负的 `v_xor` 并进乘法的 neg 修饰位 | +0.54 | 2222 |
| 2 | cyc03h | mem2 里 inv、v、β 读地址的 `v_add_u32` 提到区域开头 | −0.88 | 2116 |
| 3 | cyc03hm1spread | mem1 先发 LDS 读，4 组 copy 分散到读之间（每 4 条 kr 读跟一组） | −1.17 | 2070 |
| 4 | s_z | 删掉 p0 的 11 条 `v_add_u32 v, 0, v`（`+0` 拷贝） | −1.88 | 1962 |
| 5 | s_zl | 回边的 12 条 SALU 挪到最后一个 `s_barrier` 之前 | −2.55 | 1897 |
| 6 | d2s0 | dot2 的 32 条 pk 全部提前，只等一次 LDS，之后 MFMA 连发 | −3.81 | 1793 |
| 7 | gt3k2 | p1～p3 的 gt 前 2 条读挪到上一个 chunk 的 mem2 | 约 −4.1 | 1756 |
| 8 | b1n2 | β copy 的地址计算挪进 dot1 的 `s_nop` | 约 −4.3 | 1732 |
| 9 | b1n2s3 | dot2 B 段的 MFMA 阴影里补发 3 个 A tile 的乘法（pk 拆成 `v_mul_f32`） | 约 −4.5 | 1704 |
| 10 | b1n2s4t1 | dot1 的 MFMA 结果写回 srcC 的寄存器（tie），去掉寄存器分配造成的 `s_nop`；第 9 步改成 4 个 A tile | 约 −4.8 | 1663 |

- µs 是 T = 16384 上相对正常编译的累计变化，来自不同批次，第 7～10 步是估计值。同一时段、同一张卡上直接比，b1n2s4t1 比正常编译快 4.3～4.7 µs。
- 第 1 步单独做是变慢的，和第 2 步一起才有收益。

### 1.3 记号

- **pk：** `v_pk_mul_f32`，一条指令算 2 个 f32。
- **srcA、srcB、srcC：** MFMA 的三个输入，结果是 A·B + C。
- **MFMA 的阴影：** 一条 16x16x32 的 MFMA 发出后，矩阵单元要算约 16 个周期，这段时间里同一个 wave 还能发别的指令。
- **等待状态（wait state）：** 硬件要求两条指令之间空出的间隔。`s_nop N` 提供 N + 1 个，中间每条普通指令算 1 个。
- **piece：** 研究 kernel 和 aiter kernel 都把 g_b、g_a 等状态按行切成 4 块，每块 32 行，一块叫一个 piece（4.3 节）。
- **区域：** 主循环里两个 `s_barrier` 之间的代码，每次迭代 17 个（1.1 节）。
- **段：** 两个 `sched_barrier` 之间的代码。machine scheduler 只在段里调整顺序，不把指令移出段。
- **槽：** 两组 wave 错开一个 cluster，所以两个 `s_barrier` 之间是一组的 dot cluster 和另一组的 mem cluster 一起跑，这段时间叫一个槽。X 是 sl0 的 dot1 加 sl1 的 mem2，Y 是 sl0 的 mem1 加 sl1 的 dot1，Z 是 sl0 的 dot2 加 sl1 的 mem1，W 是 sl0 的 mem2 加 sl1 的 dot2。下标是位置，例如 Y_1 是 p1 的 Y 槽。
- **ATT 每 chunk：** 用 ATT 抓的主循环周期，折算到一个 chunk。
- **指令串：** 一条指令一个字母，从左到右是发射顺序：

| 字母 | 指令 | 字母 | 指令 |
| --- | --- | --- | --- |
| `M` | bf16 MFMA | `r` | LDS 读（`ds_read*`） |
| `I` | f16 MFMA | `g` | global → LDS 的 copy（`buffer_load … lds`） |
| `p` | `v_pk_mul_f32` | `w` | `s_waitcnt` |
| `m` | `v_mul_f32` | `n` | `s_nop` |
| `b`、`h` | 转成 bf16、f16 的 cvt | `s` | 其他 SALU，包括 `s_barrier` |
| `x` | `v_xor` | `.` | 其他 VALU |
| `f` | bf16 转 f32 | | |

## 2. 10 步逐项对照

以上游 Triton 为准。上游上的研究 kernel 是 `k2_ab_fused_wp2_pipe4g`，即 `k2_ab_fused_wp2_pipe4u` 加上一组 constexpr 开关。开关为 0 表示 `k2_ab_fused_wp2_pipe4u` 原来的顺序；一个模式名对应一组开关（读法见 5.2 节）。"相对 `g_lt`"是在 `g_lt` 上再开这个开关（同一批，GPU 2，10 轮中位数）；第 1、9 步的单条 MFMA 写法（`D1=3`、`D2=4`）是 GPU 2、3 各 5 次的平均（7.1 节）。

| 步 | MIR 的改动 | 源码写法 | 结果 | 结论 |
| --- | --- | --- | --- | --- |
| 1 | dot1 逐条重排 | `D1=3`：一条 MFMA 一段，顺序和 MIR 基本相同（`g_d1t`）；按 piece 排的 `D1=1/2` | `D1=3` 慢 0.6～0.8 µs；`D1=1/2` 慢 1.12 / 0.69 µs | 能写，更慢，7.1 |
| 2 | mem2 的地址 add 提前 | `M2=1/2`：先读 inv、v、β，再读 kd | 慢 1.28 / 0.60 µs | 写不出来，7.2 |
| 3 前一半 | mem1 先读后 copy | `M1=3`：先读 kr，再读 gt，最后发 copy | 两张卡都快 0.7 µs（复测 5 次；累加表那次 GPU 3 快 0.9、GPU 2 快 0.2） | 落地，5.2 |
| 3 后一半 | copy 分散到读之间 | `M1=4`：kr 在 LDS 里切成 4 段，每段读完跟一组 copy | 慢 0.95 µs | 能写，没有收益，5.4 |
| 4 | 删 `+0` 拷贝 | 没有写法；`disable-cgp` 能去掉，但 spill | 195 µs | 写不出来，7.3 |
| 5 | 回边 SALU 提前 | `LT=1`：copy 的基址写成循环携带的值，在 mem2 p3 推进 | 回边 12 → 4 条，两张卡都快约 1.1 µs | 落地，6.1 |
| 6 | dot2 的 pk 全部提前 | `D2=1` | 和 MIR 同序，VGPR 221 → 229 | 落地，第 4 节 |
| 7 | gt 的读提前 | `GTE=1/2`：mem2 先读下一个 chunk 的前 1～2 个 gt piece | −0.4～+0.9 µs，不同批次方向相反 | 能写，没有收益 |
| 8 | β clamp 挪进 dot1 | `BD1=2` | ATT 每 chunk 2060 → 2008 | 落地，6.2 |
| 9 | dot2 的 B 段补发 A 的乘法 | `D2=4`、`D2N=N`：一条 MFMA 一段，最后 N 个 A tile 的 `g·gt` 对半拆，跟在 B MFMA 后面，指令串和 MIR 一字不差（`g_t3`、`g_t4`）；按 piece 排的 `D2S=N` | `D2N=3/4` 慢 0.4～0.5 / 1.0～1.5 µs，`D2N=0`（`g_t0`）持平；`D2S=3` 慢 0.90 µs | 能写，更慢，7.1 |
| 10 | MFMA tie | 没有写法 | — | 写不出来，7.4 |

**逐项累加：** T = 16384，GPU 2、3 同时跑，20 轮中位数。每一行在上一行的基础上多开一项。

| 模式 | 新开的开关 | GPU 2（µs） | GPU 3（µs） | VGPR | 回边 SALU | ATT 每 chunk |
| --- | --- | --- | --- | --- | --- | --- |
| `g_base` | 上游编译原版 | 50.81 | 49.25 | 221 | 12 | 2156 |
| `g_p` | 拆成 piece | 50.34 | 49.41 | 221 | 12 | 2136 |
| `g_d2` | `D2=1` | 50.32 | 48.86 | 229 | 12 | 2144 |
| `g_d2m3` | `M1=3` | 50.08 | 47.93 | 229 | 12 | 2060 |
| `g_d2b2m3` | `BD1=2` | 49.50 | 48.18 | 229 | 12 | 2008 |
| **`g_lt`** | `LT=1` | **48.35** | **47.06** | 229 | 4 | 1978 |
| ROCm 正常编译 | — | 50.37 / 50.39 | 49.59 / 50.27 | 225 | 12 | — |
| ROCm + MIR b1n2s4t1 | — | 46.14 / 46.07 | 45.03 / 45.44 | 207 | 0 | 1663 |

- 上游各行的误差都是 (5.43e-04, 1.88e-03)，ROCm 两行都是 (5.59e-04, 1.90e-03)，都没有 spill；编译器不同，误差略有差别。ROCm 两行是在同一时段交替跑的两次。
- 同一个模式两次运行能差 0.3～0.5 µs，中间几行的单步差不必细抠（GPU 3 上 `g_d2b2m3` 还比 `g_d2m3` 慢）；首尾两行在两张卡上一致。
- ATT 是在 GPU 2 上抓的，`g_lt` 和 MIR b1n2s4t1 是两次的平均，其余各一次。
- 这张表是在 f8cddb947b 上测的。最新的 173580104f 上，这几个模式编出来的 ISA 逐字节相同。在它上面复测（GPU 2、3 各 5 次，一次只跑一个进程），各模式和表里差 0.5 µs 以内，`g_lt` 比 `g_base` 快 2.16、2.01 µs；ATT 每 chunk `g_base` 2156、`g_lt` 1976。

**其他试过但没用的：** 相对 `g_lt`（同一批，GPU 2，10 轮中位数）。

| 模式 | 开关或选项 | 相对 `g_lt`（µs） | 备注 |
| --- | --- | --- | --- |
| `g_ltms` | `schedule_hint="mfma-schedule"` | +0.16 | ISA 与 `g_lt` 字节相同。它在两个 sched barrier 之间把 MFMA 和访存交错排，以访存为锚点，K2 的 dot cluster 里没有访存 |
| `g_ltilp` | `llvm_fn_attrs="amdgpu-sched-strategy=max-ilp"` | +1.16 | |
| `g_ltiilp` | `…=iterative-ilp` | +0.58 | VGPR 210 |
| `g_ltmc` | `…=max-memory-clause` | −0.02 | |
| `g_ltl` | `llvm_cl_opts="disable-lsr"` | +0.41 | 见 6.1 |
| `g_vp` | `VP=1`：p3 的 v 也在 mem2 里转 f32 | +0.13 | |
| `g_ltvd` | build 选项 `v_cvt_dot1=True`：4 个位置的 v 都在 dot1 里转 f32 | +0.39 | |

## 3. aiter 里的 kernel

### 3.1 结构

`flash_kda_k2.py` 里的 `k2_ab_fused_pipe_gluon` 按 1.1 节的主循环做 pass A。

- 一个 program 算一个 (segment, head)，持有两条链完整的 [K, V] = [128, 128] 状态。grid 是 (1, segment 数 × H)，8 个 warp，`waves_per_eu=2`。状态取负保存（g = −h）。
- 每个 chunk 是 4 个 warp-pipeline stage：dot1、mem1、dot2、mem2。chunk 拷进 LDS 里一个有 4 个缓冲位的环，chunk j 计算时，chunk j + 1～j + 3 的 copy 都在路上。每个 chunk 有两个 commit group：kd、inv、v、β（在 mem1 发），kr、gt（在 mem2 发）。
- chunk j 用第 (j − n mod 4) mod 4 个缓冲位，n 是这个 segment 的 chunk 数。前 n mod 4 个 chunk 在一个循环里跑，缓冲位在运行时才确定；其余 chunk 每 4 个一组，缓冲位是编译期常数（p0～p3），最后一组展开成直线代码。n < 4 时，越过末尾的 copy 重复拷 chunk n − 1，这样 wait 的计数仍然正确。
- 各 segment 的长度可以不同，最后一个 chunk 可以不满。只有 chunk n − 1 的 copy 对越过 segment 末尾的行加 mask：v 和 β 的这些行读成 0。kd、inv、kr、gt 来自 K1 的 workspace。
- kernel 里有第 2 节的第 6 步（`_dot2`，`D2=1` 的顺序）、第 3 步的前一半（`_step` 的 mem1，`M1=3` 的顺序）和第 5 步（`pk` 和 `_advance`，`LT=1` 的写法）。
- 发射时设 `llvm_fn_attrs="target-features=-load-store-opt"`（3.2 节）。

**启用条件和回退：** Triton 有 `gl.amd.hint.sched_barrier` 和 buffer descriptor 接口（`gl.amd.cdna4.buffer`）时，`PASS_A_PIPELINED` 为真，`k2_ab_fused_fast` 发射 `k2_ab_fused_pipe_gluon`；否则发射 `k2_ab_fused_gluon`，行为不变。

- wrapper `gluon_k2_ab_fused` 的签名不变。走流水线路径时调度是固定的，wrapper 不用 `BW`、`num_warps` 和 `waves_per_eu`。(C, K, V) 不是 (32, 128, 128) 时抛 `ValueError`；分发逻辑（`flash_kda.py` 的 `_gluon_k2_usable`）只在这组值下选 Gluon K2。
- 只有分段调度（`flash_kda.py` 的 `_choose_chunks_per_seg`）把序列切成 segment 时才会跑 pass A：最长的序列有 64 个 chunk 以上（T ≥ 2048），并且不分段时的 grid 填不满 CU。`CHUNK_DELTA_ATTN_FLASH_KDA_SEG` 可以强制指定 segment 长度。

**各 Triton 版本：** 两个接口是下面两个 PR 加进 upstream main 的。kernel 用到的其他 Gluon 接口和编译选项 `llvm_fn_attrs` 都更早，upstream 3.8.0 里已经有。

- `gl.amd.hint.sched_barrier`：[triton-lang/triton#11932](https://github.com/triton-lang/triton/pull/11932)，2026-09-24。`gl.amd.hint` 模块也是这个 PR 加的。
- buffer descriptor 接口：[triton-lang/triton#12345](https://github.com/triton-lang/triton/pull/12345)，2026-10-25。kernel 用的 `make_buffer_descriptor`、`load_to_shared` 和 `store` 都是这个 PR 加的。

| Triton | 从 main 分出 | 发布 | `sched_barrier` | buffer descriptor 接口 | pass A |
| --- | --- | --- | --- | --- | --- |
| [3.8.0](https://github.com/triton-lang/triton/tree/v3.8.0) | 2026-06-23（3bb23dde71） | 2026-08-28（PyPI） | 没有 | 没有 | `k2_ab_fused_gluon` |
| 3.8.0，AMD build | 不知道 | 2026-09-17（pypi.amd.com） | 没有 | 没有 | `k2_ab_fused_gluon` |
| [3.9.0](https://github.com/triton-lang/triton/tree/v3.9.0) | 2026-10-01（aad2a60d95） | 2026-10-27（PyPI） | 有 | 没有 | `k2_ab_fused_gluon` |
| main 0df169311f | | | 有 | 有 | `k2_ab_fused_pipe_gluon` |

- 3.8 的 release 分支比两个 PR 都早，也没有 cherry-pick 它们，所以 Triton 3.8 两个都没有。
- 3.9 的 release 分支在两个 PR 之间分出，没有 cherry-pick #12345，所以 Triton 3.9.0 只有 `sched_barrier`，pass A 也是 `k2_ab_fused_gluon`。到 2026-11-13，还没有 2026-10-25 以后分出的 release 分支，所以还没有哪个 Triton 发布版两个都有。
- upstream 3.8.0 和 3.9.0 两行是按 tag 上的源码判断的，没有装来测。
- aiter CI 装的是 AMD 的 build `3.8.0+amd.rocm7.2.0.git7cb7b059`（pypi.amd.com 的 `release_0914` 目录；表里的日期是 wheel 文件的日期），不是 upstream 的 3.8.0。在这个 build 上实测，`gl.amd.hint` 和 `gl.amd.cdna4.buffer` 都没有。它的 codegen LLVM 是 8aa7100b，不是 upstream 3.8.0 钉的 LLVM。这个 LLVM 还会让 kernel spill：早先一版用旧接口、不带 `sched_barrier` 的 kernel，在这个 build 上 spill 了 122 个 VGPR，T = 16384、H = 12 时要 112 µs。

### 3.2 和研究 kernel 的差别

| 项目 | 研究 kernel（`g_lt`） | aiter kernel |
| --- | --- | --- |
| 一个 program 的 chunk | 固定是 4 的倍数，chunk 都是满的 | 每个 segment 可以是任意 n ≥ 0；最后一个 chunk 可以不满 |
| β copy | 从 [T, H] 的输入按 2C 宽读，偏移 clamp 到 T − 1（`_beta_off`，第 8 步） | 从 K1 的 workspace 按 2C 宽读，只有最后一个 chunk 的 copy 加 mask。没有 clamp，第 8 步没有东西可挪 |
| `LT=1` 循环携带的值 | 6 个 copy 基址和 β 上限（int32） | 6 个 copy 基址 |
| kd、inv 的读 | B128：K1 按读的列顺序写 kd、inv，一个 lane 用一条 `ds_read_b128` 读完自己的 16 字节 | 两条 `ds_read_b64`；K1 不改 |
| 编译器 | 研究用的编译器补丁：不用 AGPR（`amdgpu-agpr-alloc=0,0`），关掉 SILoadStoreOptimizer | `llvm_fn_attrs="target-features=-load-store-opt"`；不打补丁，上游 Triton 也不给这个 kernel 分 AGPR |
| Gluon 接口 | `buffer_load_to_shared`、`load_shared_relaxed`、`buffer_store` | 带 buffer descriptor 的 `cdna4.buffer.load_to_shared`、`cdna4.buffer.store`，以及 `smem.load` |
| layout | kernel 参数 | 模块级常量 |

- **SILoadStoreOptimizer：** 开着这个 pass 时，LLVM 把 kd、inv 第 r 行和第 r + 16 行的读合成 `ds_read2st64_b64`，之后要用 `v_mov` 重新凑出 MFMA 操作数的寄存器对。开着它，aiter kernel 的 ISA 里有 90 条 `ds_read2st64_b64`。GPU 3 上 event 计时：T = 16384、H = 12，开着 56.27 µs，关掉 44.58 µs；T = 4096、H = 12，21.79 µs 和 18.71 µs。
- **B128：** T = 16384、H = 12 时 aiter kernel 是 46.42 µs（graph replay），研究的 `g_lt` 在同一张卡上、另一次运行里是 47.06 µs。两次运行能差 0.3～1.3 µs（3.4 节），所以这里测不出 B128 的作用，K1 可以不动。
- **layout：** `fast_launch` 每次发射都用全部关键字参数生成 cache key。12 个 layout 作为参数时，对 layout 对象求 hash 约 5.5 µs，一次发射的主机耗时约 27.5 µs。改成模块级常量后，layout 经 repr 进入 JIT 的 cache key，一次发射约 13.5 µs。
- **Gluon 接口：** Triton main 上，`load_shared_relaxed` 从 2026-10-26 起废弃（[triton-lang/triton#12478](https://github.com/triton-lang/triton/pull/12478)），基于指针的 buffer 接口从 2026-10-27 起废弃（[triton-lang/triton#12527](https://github.com/triton-lang/triton/pull/12527)）。aiter kernel 用的是替代接口。每个 descriptor 的 `num_records = 2**31 - 2`，即旧的指针接口用的上界；访问不越界靠 mask 保证。时间和旧接口相同（A/B 差 ±0.45 µs 以内）。换成 descriptor 接口后，有 12～14 个 SGPR spill 到 VGPR 的 lane（`v_writelane`、`v_readlane`），都不在循环里。

### 3.3 结果

MI350X（GPU 3），Triton main 0df169311f，只测 pass A。"旧"是带调好配置的 `k2_ab_fused_gluon`。eager 计时期间 KFD 驱逐计数为 0。

**GPU 时间（CUDA graph replay）和每次发射的主机耗时：**

| T，H（B = 1） | segment 数 × chunk 数 | GPU，新（µs） | GPU，旧（µs） | 加速 | 主机，新（µs） | 主机，旧（µs） |
| --- | --- | --- | --- | --- | --- | --- |
| 2048，12 | 16 × 4 | 14.13 | 18.09 | 1.28× | 13.58 | 25.57 |
| 4096，12 | 16 × 8 | 18.49 | 25.03 | 1.35× | 13.28 | 25.37 |
| 2048，32 | 8 × 8 | 21.54 | 26.83 | 1.25× | 13.45 | 25.60 |
| 8192，12 | 16 × 16 | 27.65 | 40.04 | 1.45× | 13.44 | 25.32 |
| 8192，16 | 16 × 16 | 32.00 | 43.61 | 1.36× | 13.51 | 25.80 |
| 16384，12 | 16 × 32 | 46.42 | 73.64 | 1.59× | 13.40 | 25.33 |
| 32768，8 | 16 × 64 | 74.09 | 124.18 | 1.68× | 13.46 | 25.41 |
| 65536，4 | 32 × 64 | 74.07 | 124.25 | 1.68× | 13.43 | 25.97 |
| 4096，64 | 4 × 32 | 58.97 | 82.26 | 1.40× | 14.07 | 26.43 |

**eager 发射**（连续发射、event 计时，中位数）：

| 输入 | segment 数 × chunk 数 | 新（µs） | 旧（µs） | 加速 |
| --- | --- | --- | --- | --- |
| T = 2048，H = 4 | 16 × 4 | 13.65 | 25.77 | 1.89× |
| T = 2048，H = 12 | 16 × 4 | 13.77 | 25.70 | 1.87× |
| T = 2048，H = 32 | 8 × 8 | 21.13 | 27.39 | 1.30× |
| T = 4096，H = 4 | 16 × 8 | 14.68 | 25.96 | 1.77× |
| T = 4096，H = 8 | 16 × 8 | 16.38 | 26.19 | 1.60× |
| T = 4096，H = 12 | 16 × 8 | 18.33 | 26.25 | 1.43× |
| T = 4096，H = 32 | 8 × 16 | 30.86 | 43.92 | 1.42× |
| T = 4096，H = 64 | 4 × 32 | 57.56 | 84.41 | 1.47× |
| T = 6144，H = 12 | 12 × 16 | 25.22 | 38.05 | 1.51× |
| T = 8192，H = 12 | 16 × 16 | 26.73 | 39.46 | 1.48× |
| T = 8192，H = 16 | 16 × 16 | 30.95 | 43.84 | 1.42× |
| T = 8192，H = 64 | 4 × 64 | 122.29 | 171.64 | 1.40× |
| T = 16384，H = 12 | 16 × 32 | 44.88 | 73.80 | 1.64× |
| T = 32768，H = 8 | 16 × 64 | 74.56 | 124.66 | 1.67× |
| T = 65536，H = 4 | 32 × 64 | 73.81 | 124.42 | 1.69× |
| B = 4，T = 4096，H = 8 | 32 × 16 | 30.98 | 43.44 | 1.40× |
| 变长 5000、3001、8191、777，H = 8 | 17 × 25～32 | 40.60 | 67.97 | 1.67× |

- chunk 少时，eager 发射受主机耗时限制，时间就是一次发射的主机耗时，所以 T = 2048 的 eager 加速比 GPU 加速大。
- 在 Triton 3.8.0 上，pass A 仍是 `k2_ab_fused_gluon`，主机和 GPU 时间都不变。`k2_ab_fused_gluon` 在 Triton 3.8.0 上比在 Triton main 上快：T = 16384、H = 12 是 67.39 µs（graph replay），T = 65536、H = 4 是 120.96 µs。和这两个时间比，Triton main 上的新 kernel 快 1.45 倍和 1.63 倍。
- 寄存器：239 或 240 个 VGPR，没有 AGPR，没有 VGPR spill，occupancy 2（测试和 benchmark 编出来的 17 个变体都是这样）。
- 和 fp64 参考比，误差和 `k2_ab_fused_gluon` 相同。

### 3.4 怎么测试和计时

**单元测试**（Triton main 或 Triton 3.8.0）：

```bash
python3 -m pytest op_tests/triton_tests/kimi_delta_attn/test_flash_kda.py \
  op_tests/triton_tests/kimi_delta_attn/test_fast_launch.py -q -rs
```

- Triton main 0df169311f：116 passed，1 skipped（跳过的测试要第二张卡）。Triton 3.8.0 上相同。
- 测试经 `flash_kda_fwd` 跑 pass A：分段、segment 长短不一的变长输入、不满的最后一个 chunk，以及弱 gate（`lower_bound` 为 −1、−0.1、−0.01）。默认的下界 −5 会让 gt 约为 1e-35，A 链全是 0；弱 gate 的测试检查的就是 A 链。
- `flash_kda_k2.py` 的 `PASS_A_PIPELINED` 表示 pass A 用的是哪个 kernel。`flash_kda.py` 打印的 info 日志 `flash_kda route: ...` 给出路由，每种新路由打印一次。

**计时：**

- 在同一个进程、同一张卡上交替比较两个 kernel。同一个 kernel 两次运行能差 0.3～1.3 µs，不同 GPU 之间能差 1 µs 以上。
- CUDA graph replay 测的是 GPU 时间。eager 连续发射还包含主机耗时：chunk 少时，一次发射受主机限制。
- 一次只跑一个 GPU 进程，并盯着 KFD 的驱逐计数：进程运行时，`/sys/class/kfd/kfd/proc/<host 上的 pid>/stats_*/evicted_ms` 应当一直是 0。2026-11-09 一轮 CI 卡住时，所有 GPU 进程的队列 90% 以上的时间被驱逐，新建 stream、捕获 graph 要几十秒。两个进程同时跑也会互相驱逐。
- 不同版本的 Triton 用不同的 `TRITON_CACHE_DIR`。Triton 的 cache key 不包含所有影响编译的环境变量；对比编译选项时，每个变体用自己的缓存目录，或者设 `TRITON_ALWAYS_COMPILE=1`。
- 看 ISA：读 Triton 缓存里的 `.amdgcn` 文件，检查 `.vgpr_count`、`.vgpr_spill_count`、`.sgpr_spill_count` 和 `ds_read2st64`（应当一条也没有）。

### 3.5 限制和待办

- **CI 覆盖：** aiter CI 用 Triton 3.8.0，CI 只测到回退路径。要等 CI 的 Triton 有 `gl.amd.hint.sched_barrier` 和 `gl.amd.cdna4.buffer`，CI 才会测到新 kernel。升到 Triton 3.9.0 也不够，它没有 buffer descriptor 接口（3.1 节）。
- **没测过的 Triton 版本：** kernel 在 Triton main 0df169311f 上测过。2026-10-25（buffer descriptor 接口，[triton-lang/triton#12345](https://github.com/triton-lang/triton/pull/12345)）到 2026-10-27（codegen LLVM 08b05a33，[triton-lang/triton#12496](https://github.com/triton-lang/triton/pull/12496)）之间的 Triton main 能通过启用条件，但用的是更老的 codegen LLVM，这些版本没测过。Triton 3.8 说明，更老的 codegen LLVM 可能让这个 kernel spill。
- **没有专门的单元测试和 benchmark：** pass A 由 `flash_kda_fwd` 的测试覆盖（3.4 节）。
- **其他 K2 kernel 的主机耗时：** `k2_c_gluon` 和回退用的 `k2_ab_fused_gluon` 仍然把 layout 当 kernel 参数传。`k2_ab_fused_gluon` 一次发射的主机耗时约 25.5 µs，改成模块级常量（3.2 节）大约能减半。`k2_c_gluon` 的主机耗时没有测。
- **其他 kernel 里的旧接口：** `k2_ab_fused_gluon` 和 `k2_c_gluon` 还在用已废弃的接口，因为它们要能在 Triton 3.8 上跑。
- **调好的配置：** `aiter/ops/triton/configs/gfx950/gluon/attention/chunk_delta_attn/DEFAULT.json` 里的 `k2_ab_fused_gluon_wide`、`k2_ab_fused_gluon_narrow` 只对回退路径生效。
- **`sched_barrier` 的 mask：** `gl.amd.hint.sched_barrier()` 只接受 mask 0，Gluon 也没有 `sched_group_barrier`（7.1 节）。
- **长度为 0 的序列：** `prepare_chunk_indices`（`kimi_delta_attn/utils/index.py`）按 chunk 序号为 0 的位置推出每个 chunk 的序列号。长度为 0 的序列没有 chunk，它后面的 chunk 都会被算到前一个序列号上：`cu_seqlens = [0, 64, 64, 128]`、C = 32 时得到 `[0, 0, 1, 1]`，应当是 `[0, 0, 2, 2]`。新旧 pass A 拿到的是同样错误的输入。这个问题在本次改动之前的 main 里就有，本次改动没有碰它。

## 4. dot2 的顺序（第 6 步）

对应 MIR 第 6 步。上游用 `sched_barrier` 写（`D2=1`，4.3 节）；Triton 3.7 源码变体用 inline asm 写（`d2pk16w`，4.4 节）。aiter 里，`flash_kda_k2.py` 的 `_dot2` 就是 `D2=1` 的顺序。

### 4.1 dot2 算什么

研究 kernel 的 `_step` 里，dot2 stage 的源码是：

```python
g_b = gl.amd.cdna4.mfma(kr, big_b_op, g_b * gt[:, None])
if FULL_COMPUTE:
    g_a = gl.amd.cdna4.mfma(kr, big_a_op, g_a * gt[:, None])
gb_op = gl.convert_layout(g_b.to(gl.bfloat16), B_OP)
```

- g_b、g_a 各有 8 个 16x16 的输出 tile，下面叫 B0～B7、A0～A7。每个 tile 一条 `v_mfma_f32_16x16x32_bf16`，共 16 条 MFMA。
- srcC 是 `g * gt`。每个线程在每个 tile 上有 4 个 f32，要 2 条 pk；两条链共 32 条 pk。
- gt 是 mem1 里 8 条 `ds_read_b128` 读进来的，B_j 和 A_j 用同一段 gt。
- 最后 16 条 cvt 把 g_b 转成 bf16，给下一个 chunk 的 dot1 用。

### 4.2 目标顺序

区域 6（dot2 p1）的指令串：

```text
MIR 第 6 步     wppppppppppppppppppppppppppppppppMMMMMMMMMbbMbbMbbMbbMbbMbbMbbMbbws
上游，g_p       ppppMmmpppppppppppMmmppMmmppMmmppMmmppMmmpbMbbbMbbbppMbbbMbbbMbbbMMMMMws
ROCm 正常编译   wppwppwppwppwppwppwppwppppppMmmppMmmMMMMMnmmpMmmpMnmmpMnmmpMMnbbbMbbbMbbbMbbbMbbbbws
```

MIR 第 6 步把 32 条 pk 全部提前。第一条 pk 用的正是最后读回的 gt，所以 `SIInsertWaitcnts` 只插一条 `lgkmcnt(0)`；之后 MFMA 连发，中间只夹 cvt。编译器自己排的顺序有这几个问题：

1. **pk 被拆开。** MFMA 没算完时发 packed f32 要停约 16 个周期，标量的 `v_mul_f32` 不停，所以 LLVM 的 `SIPreEmitPeephole` 会把落在 MFMA 阴影里的 pk 拆成两条 `v_mul_f32`（指令串里的 `mm`），多出指令。上游 `g_p` 每个 dot2 拆了 6 条；ROCm 正常编译 4 个 dot2 共拆了 28 条，多了 28 条指令。
2. **MFMA 的阴影里有 VALU。** MIR 第 6 步的实验里，dot2 的阴影空着比塞 VALU 快：一组 wave 在阴影里发 VALU，会拖慢同一个 SIMD 上另一组 wave 的 mem cluster。
3. **ROCm 正常编译还等了 8 次 LDS。** 开头是 8 组 `wpp`：每读回一段 gt 就等一次，算 2 条 pk。上游编译在 mem1 末尾、`s_barrier` 之前就等完了 LDS（`s_waitcnt lgkmcnt(0)`），dot2 开头不用再等。

### 4.3 `D2=1`

**先把状态拆成 piece。** `sched_barrier` 只能放在两个 op 之间。原版 dot2 的 `mfma(kr, big_b_op, g_b * gt)` 一次出 8 个输出 tile，就是 8 条 MFMA，中间插不进 `sched_barrier`。研究 kernel 让 g_b、g_a 和 gb_op（g_b 转成的 bf16 B 操作数）在整个循环里都是 4 个 32 行的 piece（一个 tuple）。一个 piece 是 dot2 的两个输出 tile，也正好是下一个 dot1 的一个 K 步。kr 从 LDS 读出来以后，用 `gl.amd.slice` 在寄存器里按行切成 4 个 piece（`krp`），不生成指令：

```python
@gluon.jit
def _sb(ON: gl.constexpr):
    if ON:
        gl.amd.hint.sched_barrier()


@gluon.jit
def _rows(x, i: gl.constexpr, R: gl.constexpr):
    return gl.amd.slice(x, [R, x.shape[1]], [i * R, 0])
```

只拆 piece、不加 `sched_barrier`（`g_p`）时，指令条数和 VGPR 都不变，时间在噪声内。

**`_dot2` 里 `D2=1` 的部分**（省略了 `D2S`）：先算全部 `g * gt`，再连发 4 个 B piece 的 MFMA，最后 A piece 的 MFMA 和 B 的 cvt 交替，每对一段：

```python
cb0 = g_b[0] * gt[0][:, None]          # cb1～cb3、ca0～ca3 同样：32 条 pk
...
_sb(True)
nb0 = gl.amd.cdna4.mfma(krp[0], big_b_op, cb0)
nb1 = gl.amd.cdna4.mfma(krp[1], big_b_op, cb1)
nb2 = gl.amd.cdna4.mfma(krp[2], big_b_op, cb2)
nb3 = gl.amd.cdna4.mfma(krp[3], big_b_op, cb3)
_sb(True)
na0 = gl.amd.cdna4.mfma(krp[0], big_a_op, ca0)
ob0 = _bop(nb0, B_OP)                  # g_b 转 bf16，作下一个 dot1 的 B 操作数
_sb(True)
na1 = gl.amd.cdna4.mfma(krp[1], big_a_op, ca1)
ob1 = _bop(nb1, B_OP)
_sb(True)
...                                    # (na2, ob2)、(na3, ob3) 同样
```

**结果：**

```text
上游，g_d2      ppppppppppppppppppppppppppppppppMMMMMMMMMbbbMbMbbbMbMbbbMbMbbbMbws
MIR 第 6 步     wppppppppppppppppppppppppppppppppMMMMMMMMMbbMbbMbbMbbMbbMbbMbbMbbws
```

- 32 条 pk 在前、9 条 MFMA 连发，和 MIR 第 6 步一样。pk 不在 MFMA 的阴影里，不再被拆，区域 6 从 72 条指令降到 66 条。VGPR 221 → 229。
- 单开 `D2=1`（`g_p` → `g_d2`）：GPU 3 快 0.55 µs，GPU 2 不变，在噪声附近。ATT 里 Z 槽（sl0 的 dot2 加 sl1 的 mem1）从 537 涨到 616 周期，再开 `M1=3`、`BD1=2` 后降到 464。原因应该是：dot2 开头 32 条 pk 连发时，同一个 SIMD 上另一组 wave 的 mem1 要发的 VALU（gt 地址、β clamp）发不出去；`M1=3` 让 kr 读先发，`BD1=2` 把 β clamp 挪走。这一点没有逐条看 ATT。

### 4.4 inline asm 写法：`d2pk16w`（Triton 3.7 源码变体）

把 g_b、g_a 的 `g * gt` 各写成一段 16 条 `v_pk_mul_f32` 的 non-pure inline asm。asm 是一个整体，scheduler 不会拆开它，也不会往中间插 MFMA；代价是要自己处理 asm 和外面 MFMA 之间的 hazard。在 ROCm Triton 3.7、GPU 3 上：48.17～48.49 µs，比正常编译（49.23～49.35 µs）快约 0.9 µs，VGPR 225 → 227，没有 spill，误差不变。

```text
d2pk16w         wppppppppppppppppnnMMMMMMMMnppppppppppppppppnnMMMbbbMbbbMbbbMbbbMbbbMbws
```

和 `D2=1` 比：A 的 16 条 pk 夹在 B 和 A 的 MFMA 之间（`D2=1` 是 32 条 pk 全在前面），每个 dot2 多 5 条 `s_nop`；VGPR 只多 2 个（`D2=1` 多 8 个）。

#### 源码

两个开关在文件开头：

```python
D2_PK16 = gl.constexpr("d2pk16" in _SRC or "d2pk16w" in _SRC)
D2_WAR = gl.constexpr("d2pk16w" in _SRC)
```

dot2 stage 里的调用：

```python
with _stage("dot2", PRIO_DOT):
    if D2_PK16 and FULL_COMPUTE:
        gb_g = _gate_pk16n(g_b, gt)
        ga_g = _gate_pk16n(g_a, gt, D2_WAR)
        g_b = gl.amd.cdna4.mfma(kr, big_b_op, gb_g)
        g_a = gl.amd.cdna4.mfma(kr, big_a_op, ga_g)
    ...
    gb_op = gl.convert_layout(g_b.to(gl.bfloat16), B_OP)
```

第一段算 `g_b * gt`，第二段算 `g_a * gt`；`d2pk16w` 时第二段开头多一条 `s_nop 2`。

inline asm 的模板和约束：

```python
_PK16N_ASM = gl.constexpr("\n".join(
    [f"v_pk_mul_f32 ${i}, ${16 + i}, ${32 + i}" for i in range(16)] + ["s_nop 1"]))
_PK16W_ASM = gl.constexpr("s_nop 2\n" + _PK16N_ASM.value)
_PK16N_CON = gl.constexpr(",".join(["=&v"] * 16 + ["v"] * 32))
```

展开后 `_PK16N_ASM` 是：

```text
v_pk_mul_f32 $0, $16, $32
v_pk_mul_f32 $1, $17, $33
...
v_pk_mul_f32 $15, $31, $47
s_nop 1
```

- `$0`～`$15` 是 16 个输出，约束 `=&v` 表示 VGPR、early-clobber。
- `$16`～`$31` 是 g 的 16 对寄存器，`$32`～`$47` 是广播后的 gt 的 16 对。
- 每个操作数都是一个 `int64`，也就是一对相邻的 VGPR，正好是 pk 的一个 64 位操作数。
- `_PK16W_ASM` 只是在开头多一条 `s_nop 2`。

`_gate_pk16n`：

```python
@gluon.jit
def _gate_pk16n(g, gt, WAR: gl.constexpr = False):
    # _gate_pk16 plus the MFMA hazards the asm is invisible to: s_nop 1 after
    # (VALU write -> MFMA SrcC read) and, with WAR, s_nop 2 before (the outputs
    # may reuse SrcC registers of a just-issued MFMA; LLVM requires 3 wait states).
    gtb, g = gl.broadcast(gt[:, None], g)
    if WAR:
        rp = gl.inline_asm_elementwise(_PK16W_ASM, _PK16N_CON, [_pairs(g), _pairs(gtb)],
                                       dtype=gl.int64, is_pure=False, pack=16)
    else:
        rp = gl.inline_asm_elementwise(_PK16N_ASM, _PK16N_CON, [_pairs(g), _pairs(gtb)],
                                       dtype=gl.int64, is_pure=False, pack=16)
    return _unpairs(rp, g.type.layout)
```

- `gl.broadcast` 把 gt 扩成和 g 一样的形状。
- `pack=16`：一个 asm 实例处理 16 个元素。每个线程的 g 正好是 16 个 int64（32 个 f32：8 个 tile，每个 tile 4 个），所以每个线程只生成一段 asm。
- `dtype=gl.int64`：输出也是寄存器对。Triton 只会自动打包小于 32 位的元素，f32 不会，所以要自己把两个 f32 拼成一个 int64。

`_pairs`、`_unpairs`、`_pk64`：

```python
@gluon.jit
def _pairs(x):
    # [K, BW] -> [K // 2, BW] int64 of (row 2m, row 2m + 1): adjacent MFMA output registers.
    K: gl.constexpr = x.shape[0]
    BW: gl.constexpr = x.shape[1]
    e, o = gl.split(gl.permute(gl.reshape(x, [K // 2, 2, BW]), [0, 2, 1]))
    return _pk64(e, o)


@gluon.jit
def _unpairs(rp, layout: gl.constexpr):
    K: gl.constexpr = 2 * rp.shape[0]
    BW: gl.constexpr = rp.shape[1]
    re = rp.to(gl.uint32).to(gl.float32, bitcast=True)
    ro = (rp >> 32).to(gl.uint32).to(gl.float32, bitcast=True)
    return gl.convert_layout(gl.reshape(gl.permute(gl.join(re, ro), [0, 2, 1]), [K, BW]), layout)


@gluon.jit
def _pk64(lo, hi):
    return ((hi.to(gl.uint32, bitcast=True).to(gl.int64) << 32)
            | lo.to(gl.uint32, bitcast=True).to(gl.int64))
```

- `_pairs` 把第 2m、2m + 1 行拼成一个 int64，偶数行在低 32 位。在 MFMA 输出的 layout 里，一个线程的这两行就在相邻的两个 VGPR 里，所以这对 VGPR 本来就是 pk 的一个操作数。
- `_unpairs` 把 int64 拆回两个 f32，再排回 `[K, BW]`。
- 这些移位、拼接和 `convert_layout` 只是换个方式解释同一组寄存器，LLVM 会全部消掉：下面的 ISA 里，asm 前后没有 `v_mov`。

#### 生成的 ISA

区域 6（dot2 p1），省略了中间相同模式的行：

```text
        s_waitcnt lgkmcnt(0)                              ; 只等一次 LDS
        v_pk_mul_f32 v[162:163], v[32:33], v[170:171]     ; 第一段 asm：g_b * gt
v_pk_mul_f32 v[164:165], v[34:35], v[172:173]
...                                                       ; 共 16 条，输出 v[162:169]、v[202:225]
v_pk_mul_f32 v[224:225], v[62:63], v[200:201]
s_nop 1                                                   ; asm 里的 RAW 保护
        s_nop 0                                           ; LLVM 补的，见下面的 s_nop 一节
        v_mfma_f32_16x16x32_bf16 v[32:35], v[96:99], v[100:103], v[162:165]   ; B0
        ...                                                                   ; B1～B6
        v_mfma_f32_16x16x32_bf16 v[60:63], v[68:71], v[100:103], v[222:225]   ; B7
        s_nop 2                                           ; 第二段 asm 的第一行：WAR 保护
v_pk_mul_f32 v[100:101], v[0:1], v[170:171]               ; g_a * gt
...                                                       ; 共 16 条，输出 v[100:103]、v[162:169]、v[202:221]
v_pk_mul_f32 v[220:221], v[30:31], v[200:201]
s_nop 1
        s_nop 0
        v_mfma_f32_16x16x32_bf16 v[0:3], v[96:99], v[64:67], v[100:103]       ; A0
        v_mfma_f32_16x16x32_bf16 v[4:7], v[92:95], v[64:67], v[162:165]       ; A1
        v_mfma_f32_16x16x32_bf16 v[8:11], v[88:91], v[64:67], v[166:169]      ; A2
        v_cvt_pk_bf16_f32 v88, v32, v33                   ; g_b 转 bf16，夹在 A 的 MFMA 之间
        ...
        s_waitcnt vmcnt(10) lgkmcnt(0)
        s_barrier
```

- LLVM 只给 inline asm 的第一行加缩进，没有缩进的行都是 asm 原文。
- 第一段只等一次 `lgkmcnt(0)`：asm 是一个整体，要等 8 段 gt 全部读回来才能开始。
- 第二段的输出复用了第一段已经用完的寄存器（v[162:169]、v[202:221]），还有 B 的 MFMA 的 srcB（v[100:103]，即 `big_b_op`，B7 之后不再用）。

#### 为什么这样写

**一段 asm 放 16 条 pk。** asm 是一个整体，带来三个效果：MFMA 插不进来；LLVM 不会把 pk 拆成 `v_mul_f32`；`SIInsertWaitcnts` 只在 asm 前面等一次 LDS。

**分两段，而不是一段 32 条。** MIR 第 6 步能把 32 条 pk 都放在前面而不多占寄存器，是因为每条 pk 都原地写回 g 自己的寄存器（`v_pk_mul_f32 v[62:63], v[62:63], v[164:165]`）：这对 g 被这条 pk 读完就没用了。inline asm 表达不了"输出 i 可以和输入 i 共用寄存器，但不能和别的输入共用"：

- 用 early-clobber 时，32 个输出（64 个 VGPR）要同时活着，还不能和任何输入重叠，VGPR 到 256 并 spill。
- 把输出 tie 到输入时（`d2asm`），寄存器分配在循环里多插了 63 条 `v_mov`。

分成两段以后，第一段的输出被 B 的 MFMA 读完就死了，第二段可以复用这些寄存器，VGPR 只多 2 个。

**early-clobber（`=&v`）。** 没有 `&` 时，LLVM 认为 asm 先读完全部输入、再写输出，所以允许输出和在这里死掉的输入共用寄存器。但这段 asm 有 16 条指令：如果输出 0 被分到 g 的第 1 对寄存器，第 0 条指令写完它，第 1 条读到的就是错的。`&` 保证输出不和任何输入共用寄存器，代价是 VGPR 225 → 227。

**`is_pure=False`。** 有副作用的 asm 不会被删除、复制或移出所在的 stage，两段的先后也不变。MFMA 仍然可以移动；machine scheduler 把 B 的 MFMA 排在了两段之间，第二段才能复用第一段的寄存器。

**int64 寄存器对。** 一个线程的第 2m、2m + 1 行本来就在相邻的两个 VGPR 里，拼成 int64 正好是 pk 的一个操作数，不需要额外的 `v_mov`。

#### `s_nop`：asm 里的 MFMA hazard 要自己处理

LLVM 的 hazard recognizer 看不到 asm 里面，而且按最坏情况把 asm 算作 0 个等待状态（`GCNHazardRecognizer.cpp` 的 `getCountedWaitStates`："Inline asm may be empty, so it adds none, which is the worst case"）。所以 asm 里的 pk 和外面的 MFMA 之间要等多少，得自己写。下表是用 ROCm 的 `llc` 对 gfx950 跑 hazard pass 测出来的（复现方法见下面的折叠块）：

| 情况 | LLVM 插的 | 等待状态 | d2pk16w 里对应的 |
| --- | --- | --- | --- |
| RAW：pk 写一个寄存器，紧接着 MFMA 把它当 srcC 读 | `S_NOP 1` | 2 | 每段末尾的 `s_nop 1` |
| WAR：MFMA 把一个寄存器当 srcC 读，紧接着 pk 写它 | `S_NOP 2` | 3 | 第二段开头的 `s_nop 2` |
| MFMA 读 srcB，紧接着 pk 写它 | 不插 | 0 | 不需要 |

- WAR 要等几个取决于 MFMA 的 pass 数。LLVM 把 16x16x32 bf16 当作 4-pass 的 XDL 指令（`GFX940_XDL4PassReadVgprVALUWarWaitStates = 3`）；2、8、16 pass 分别要 1、7、15 个。换 MFMA 形状要重查。
- 第二段写的寄存器里，离得最近的是 B6 的 srcC，中间隔着 B7、`s_nop 2` 和 14 条 pk，远多于 3 个等待状态。`s_nop 2` 是为最坏情况准备的：寄存器分配一变，第二段的第一条 pk 就可能正好写 B7 的 srcC。
- 每段末尾 LLVM 还会再补一条 `s_nop 0`。这是 gfx940 以后的 dst forwarding 规则：LLVM 假设 inline asm 的输出都有 forwarding hazard（源码注释 "Assume inline asm has dst forwarding hazard"），紧跟着读这些输出的 VALU（MFMA 也算）前面要 1 个等待状态。它不看 asm 里已经有的 `s_nop 1`，所以这条是多余的，但源码里去不掉。
- 不加 WAR 保护（`d2pk16`）也一样快，在现在的寄存器分配下结果也对，但寄存器分配一变就可能算错。
- 最早的版本按 8-pass 加了 12 个等待状态（`s_nop 7` + `s_nop 3`），比 `d2pk16` 慢约 1.8 µs。用 `llc` 查到实际只要 3 个以后，才改成现在的 `s_nop 2`。

<details>
<summary>用 <code>llc</code> 复现上表</summary>

把下面的内容存成 `t.mir`：

```text
---
name: war
tracksRegLiveness: false
body: |
  bb.0:
    $vgpr0_vgpr1_vgpr2_vgpr3 = V_MFMA_F32_16X16X32_BF16_vgprcd_e64 $vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11, $vgpr12_vgpr13_vgpr14_vgpr15, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr12_vgpr13 = V_PK_MUL_F32 8, $vgpr20_vgpr21, 8, $vgpr22_vgpr23, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    S_ENDPGM 0
...
---
name: war_srcb
tracksRegLiveness: false
body: |
  bb.0:
    $vgpr0_vgpr1_vgpr2_vgpr3 = V_MFMA_F32_16X16X32_BF16_vgprcd_e64 $vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11, $vgpr12_vgpr13_vgpr14_vgpr15, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr8_vgpr9 = V_PK_MUL_F32 8, $vgpr20_vgpr21, 8, $vgpr22_vgpr23, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    S_ENDPGM 0
...
---
name: raw
tracksRegLiveness: false
body: |
  bb.0:
    $vgpr12_vgpr13 = V_PK_MUL_F32 8, $vgpr20_vgpr21, 8, $vgpr22_vgpr23, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr0_vgpr1_vgpr2_vgpr3 = V_MFMA_F32_16X16X32_BF16_vgprcd_e64 $vgpr4_vgpr5_vgpr6_vgpr7, $vgpr8_vgpr9_vgpr10_vgpr11, $vgpr12_vgpr13_vgpr14_vgpr15, 0, 0, 0, implicit $mode, implicit $exec
    S_ENDPGM 0
...
```

然后运行：

```bash
/opt/rocm/llvm/bin/llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -run-pass=post-RA-hazard-rec \
  -verify-machineinstrs=0 t.mir -o - | grep -E "^name:|S_NOP|V_MFMA|V_PK_MUL"
```

`war` 的两条指令之间会出现 `S_NOP 2`，`raw` 的是 `S_NOP 1`，`war_srcb` 的没有。

</details>

#### 试过但没用的写法

| 写法 | 做了什么 | 结果 |
| --- | --- | --- |
| `d2pin` | 普通乘法，结果用空的 non-pure asm 钉住（`_pin2`） | pk 全被拆成 `v_mul_f32`（每个 dot2 64 条），VGPR 213，50.9 µs |
| `d2asm`，early-clobber | 32 条 pk 写在一段 asm 里 | VGPR 256，spill |
| `d2asm`，输出 tie 到输入（现在的代码） | 同上，输出原地写回 g | 单独用仍然 spill（VGPR 256，7 个）；加上 `m1spread` 后 VGPR 227、不 spill，但循环里多出 63 条 `v_mov`，每 chunk 247 条指令（`d2pk16` 是 234），速度和正常编译差不多 |
| `d2pk16` | 两段各 16 条，第二段开头没有 `s_nop 2` | 约 −1.0 µs，但不安全 |
| `d2pk16w`，12 个等待状态 | 第一个加了保护的版本（`s_nop 7` + `s_nop 3`） | 比 `d2pk16` 慢约 1.8 µs |

#### 结果

| | ROCm 正常编译 | d2pk16w | MIR 第 1～6 步 |
| --- | --- | --- | --- |
| 4 个 dot2 的指令数 | 328 | 289 | 269 |
| 其中 `s_waitcnt` | 27 | 8 | 8 |
| `v_pk_mul_f32` | 100 | 128 | 128 |
| `v_mul_f32` | 56 | 0 | 0 |
| `s_nop` | 13 | 21 | 1 |
| VGPR | 225 | 227 | 199 |
| T = 16384，GPU 3（µs） | 49.23～49.35 | 48.17～48.49 | 45.8～46.1 |

- 三者都是 64 条 MFMA、64 条 cvt。MIR 一列的 VGPR 和时间包含 MIR 第 1～6 步的全部改动，不只是 dot2。
- 误差和正常编译相同，T = 16384 和 8192 都查过。
- d2pk16w 的 21 条 `s_nop` 里有 20 条是 asm 带来的：每个 dot2 5 条，即两段末尾各一条 `s_nop 1` 加一条 LLVM 补的 `s_nop 0`，再加第二段开头的 `s_nop 2`。
- 和 `m1spread` 一起用没有额外收益。

#### 修改时要注意

- `pack=16` 假设每个线程的 g 正好是 16 个 int64。形状、warp 数或 layout 变了，要重新算 `pack`，并在 ISA 里确认每段还是 16 条 pk，两段之间还是 8 条 B 的 MFMA。
- `s_nop` 的取值依赖 MFMA 形状和 LLVM 的 hazard 规则。换 MFMA 或换编译器版本后，要用上面的 `llc` 方法重查。
- 输出必须保留 early-clobber（`=&v`）。
- 还可以改进的地方：第二段紧跟在 8 条 B 的 MFMA 后面。按 MIR 实验里的观察（MFMA 没算完时 packed f32 要停约 16 个周期），每个 dot2 里第二段的第一条 pk 可能要停一次；MIR 第 6 步和 `D2=1` 的 pk 都在 MFMA 前面，没有这次停顿。这一点还没有用 ATT 确认。

## 5. mem1 先读后 copy（第 3 步）

对应 MIR 第 3 步。前一半（LDS 读排在 copy 前面）上游用 `sched_barrier` 写（`M1=3`，5.2 节），Triton 3.7 源码变体用空的 inline asm 写（`_fence`，5.3 节）；后一半（copy 分散到读之间）见 5.4 节。aiter 里，`flash_kda_k2.py` 中 `_step` 的 mem1 就是 `M1=3` 的顺序。

### 5.1 问题

mem1 要发 24 条 LDS 读（16 条 kr、8 条 gt）和 4 组 global→LDS copy（kd、inv、v、β，各自带着地址 SALU 和 `m0`）。发射是按顺序的，一条卡住的 `buffer_load` 会挡住后面所有的 LDS 读，所以读应该先发。

```text
区域 5（mem1 p1）
  ROCm 正常编译   ssssgssssgssssgs.s.ss.sbgrrrrrrrrrrrrrrrr.rrrrrrrrbbbbbbbs
  上游，g_d2      ssssssssss.s.rrrrrrrrrrrrrrrrrrrrrrrrgss.gsssgss.sbgbbbbbbbws
  MIR 第 3 步     rrrrssssgrrrrssssgrrrrssssgrrrrssssgrrrrrrbbbbbbbbs
```

MIR 一行是 b1n2s4t1，还包含第 7 步，所以 gt 只剩 6 条读。

- ROCm 正常编译把 copy 排在了最前面。第一条 LDS 读要到 barrier 释放后约 156 个周期才发出；dot1 按 MIR 重排、变密以后，第二条 copy 之后还要卡 80～100 个周期才能发下一条。
- **这个顺序是 machine scheduler 定的。** 研究 kernel 的源码本来就是先读 kr、gt，再发 copy；machine scheduler 之前的 MIR 也是这个顺序（24 条读在前，4 组 copy 在后），是 machine scheduler 把 copy 提到了前面。它能随意调换，是因为 Triton 给这两类访存标了互不别名的 alias scope：LDS 读是 `!alias.scope !75, !noalias !49`，copy 是 `!alias.scope !49`（`!75` 是 `amdg.LocalLoads`，`!49` 是 `amdg.AsyncCopies`），两者之间没有依赖。
- 上游 Triton 编译出来本来就是先读后 copy，只是开头还有 13 条准备指令（copy 的地址等，5.2 节）。
- MIR 第 3 步先发 LDS 读，再把 4 组 copy 分散到读之间：每 4 条 kr 读跟一组，用的是读之间 wave 空闲的发射周期。

### 5.2 `M1=3`

**模式名怎么读：** 模式名列出了打开的开关。`g_` 表示用上游 Triton 编译的研究 kernel，后面每一截是一个开关：`d2` 是 `D2=1`，`m3` 是 `M1=3`，`b2` 是 `BD1=2`，`lt` 是 `LT=1`。名字里没有的开关都是 0。

- `g_d2`：只开 `D2=1`，也就是第 4 节的 dot2 顺序：先把 32 条 pk 算完，再连发 MFMA。mem1 没有改，`M1` 是 0。
- `g_d2m3`：在 `g_d2` 上再开 `M1=3`。

两个模式只差 `M1` 一个开关，所以 `g_d2` → `g_d2m3` 的变化都来自 `M1=3`。第 2 节的累加表也是一行多开一个开关：`g_d2` → `g_d2m3` → `g_d2b2m3` → `g_lt`，`g_lt` 是 4 个开关全开。

**`M1` 的值是排法的编号：** `M1` 决定 mem1 里 LDS 读和 copy 谁先谁后。一个数字对应一种排法，数字只是编号，`D2`、`BD1` 也是这样。0 是原来的写法，1～4 是试过的 4 种：

| `M1` | 排法 | mem1 p1（区域 5）的指令串 |
| --- | --- | --- |
| 0 | 不加 `sched_barrier`，由编译器排 | `ssssssssss.s.rrrrrrrrrrrrrrrrrrrrrrrrgss.gsssgss.sbgbbbbbbbws` |
| 1 | 先发 4 组 copy，再读 kr、gt。和 ROCm 正常编译的顺序一样，作对照 | `sssssssssss.gss.gsssgss.sngrrrrrrrrrrrrrrrr.rrrrrrrrbbbbbbbbws` |
| 2 | 读完 kr 发 1 组 copy，其余 3 组插在 gt 的读之间 | `rrrrrrrrrrrrrrrrsssssssss..ssng.rrssngrrssngrrrrssss.gbbbbbbbbws` |
| **3** | **先读 kr，再读 gt，最后发 4 组 copy**。MIR 第 3 步的前一半，落地的是这一种 | `rrrrrrrrrrrrrrrrssssssss.rrrrrrrrsss.gss.gsssgss.sbgbbbbbbbws` |
| 4 | kr 分 4 片读，每片后面跟 1 组 copy。MIR 第 3 步的完整顺序，见 5.4 节 | `ssssssssssrrrrgs..ssssrrrrgssrrrrgssss.rrrrg.rrrrrrrrbbbbbbbbws` |

`r` 是 LDS 读，`g` 是 copy，`s` 是 SALU，`.` 是其他 VALU，`b` 是 cvt（1.3 节）。

**代码里的 `M1 == 3`：** `_sb(ON)` 在 `ON` 为真时插一个 `sched_barrier`，为假时什么也不生成（4.3 节）。`sched_barrier` 是给 machine scheduler 的一道挡板：排指令时，任何指令都不能从挡板的一边挪到另一边。`M1` 是编译期常量，`_sb(M1 == 3)` 的意思就是"选的是第 3 种排法，就在这里插一道挡板"。`M1=3` 插两道挡板，把 mem1 分成三段。`_step_g` 的摘录：

```python
kr = gl.amd.cdna4.async_copy.load_shared_relaxed(kr_r.index(S).permute([1, 0]), A_OP)
krp = (_rows(kr, 0, R), _rows(kr, 1, R), _rows(kr, 2, R), _rows(kr, 3, R))
_sb(M1 == 3)                           # 挡板 1，上面是第 1 段：kr 的 16 条读
gt0 = _gt_piece(gt_s, S, 0, R, MMA)
gt1 = _gt_piece(gt_s, S, 1, R, MMA)
gt2 = _gt_piece(gt_s, S, 2, R, MMA)
gt3 = _gt_piece(gt_s, S, 3, R, MMA)
_sb(M1 >= 3)                           # 挡板 2，上面是第 2 段：gt 的 8 条读
# 第 3 段：big 转 bf16、β 地址的 clamp、4 组 copy
big_b_op = gl.convert_layout(big_b.to(ws_kr.dtype.element_ty), B_OP)
big_a_op = gl.convert_layout(big_a.to(ws_kr.dtype.element_ty), B_OP)
if M1 == 3 and BD1 == 0:
    boff = _beta_off(beta_o, blim, i_h32, H)
for w in gl.static_range(4):
    _ac_d_g(kd_w, inv_w, v_w, beta_s, pd, S3, kd_off, inv_off, v_off, boff, w)
```

- 4 组 copy 都在第 3 段，machine scheduler 没法把它们提到读的前面。
- β 地址的 clamp（`_beta_off`）在 `M1=0` 时放在 mem1 的开头，`M1=3` 把它挪到第 3 段。
- 挡板 2 写成 `M1 >= 3`，是因为 `M1=4` 也要它：`M1=4` 在每片 kr 后面有自己的挡板，读完 gt 以后同样要和 cvt 隔开。
- 摘录省略了第 7 步的 `GTE` 和 `COPY` 的分支。`COPY` 在最后 3 个 chunk 为假，这时已经没有要预取的数据。
- aiter 里的读用 `smem.load(layout)`，因为 `load_shared_relaxed` 在 Triton main 上已废弃；aiter 也没有 β clamp（3.2 节）。

**ISA 的变化**（区域 5，上表 0、3 两行）：

- `g_d2`：barrier 之后先发 13 条准备指令，第 14 条才是第一条 LDS 读。这 13 条是 copy 的地址和 `m0`（10 条 SALU）、β 地址的 clamp（2 条）、gt 的读地址（1 条），其中 `v_add_u32`（gt 的读地址）和 `v_min_i32`（β clamp）是 VALU。
- `g_d2m3`：barrier 之后第一条就是 kr 读，16 条连着发。copy 的地址和 gt 的读地址挪到 kr 读和 gt 读之间，`m0` 和 β clamp 挪到 gt 读之后。
- 两者都是 61 条指令、229 个 VGPR，只是顺序不同。

**为什么快：** 问题出在 mem1 和另一组 wave 的 dot2 一起跑的时候（Z 槽，1.3 节）。`D2=1` 以后，dot2 一开头连发 32 条 pk。下面是 ATT 里 Z 槽 p1、SIMD 0 上两组 wave 发指令的时间，周期从 barrier 放开算起（`g_d2` 是 173580104f 上复测时抓的，`g_d2m3` 是第 2 节累加表那次抓的）：

```text
g_d2
  sl0（dot2）  0～124 每 4 个周期一条 pk，共 32 条；128 开始发 MFMA
  sl1（mem1）  0～36 发 10 条 SALU；40 发第一条 VALU（gt 的读地址）；
               之后 100 个周期一条指令也没发；140、144 各发 1 条；164 发第一条 kr 读
g_d2m3
  sl1（mem1）  0～116 发完 16 条 kr 读；124 发第一条 VALU；152 发第一条 gt 读
```

- `g_d2` 的 mem1 发出第一条 VALU 以后就停住了，直到另一组 wave 的 32 条 pk 发完。一个 wave 的指令按顺序发，排在后面的 24 条 LDS 读也只能一起等。
- 同样的 mem1 代码在 Y 槽里不会停。Y 槽里另一组 wave 跑的是 dot1，dot1 的 VALU 夹在 MFMA 之间，不成串，所以 mem1 的第一条 LDS 读在第 60 个周期就发出了。
- `g_d2m3` 把 kr 读放在最前面。kr 读的地址在 mem1 开始前就在寄存器里，不需要 VALU，在另一组 wave 发 pk 的这段时间里就能发完。
- 所以 ATT 里只有 Z 槽变短：第 2 节累加表那次，p1～p3 的 Z 槽 616 → 529 周期，X、Y、W 变化不到 5 个周期。

**结果：**

- 在 173580104f 上复测（GPU 2、3 各 5 次，一次只跑一个进程）：GPU 2 50.34 → 49.63 µs，GPU 3 49.10 → 48.39 µs，都快 0.7 µs。10 次里每次都是 `g_d2m3` 快，快 0.04～1.47 µs。
- 第 2 节累加表那次（f8cddb947b，两张卡同时跑）：GPU 2 50.32 → 50.08 µs，GPU 3 48.86 → 47.93 µs；ATT 每 chunk 2144 → 2060 周期。

### 5.3 inline asm 写法：`_fence`（Triton 3.7 源码变体）

Triton 3.7 源码变体用一段空的 non-pure inline asm 把读和 copy 隔开：

```python
@gluon.jit
def _fence(x):
    # Empty non-pure asm: the machine scheduler keeps memory operations on their
    # side of it (ALU instructions still move freely).
    return gl.inline_asm_elementwise("", "=s,s", [x], dtype=gl.int32, is_pure=False, pack=1)
```

- `m1late`：kr、gt 读完以后放一个 `_fence(i_h32)`，4 组 copy 排在它后面。
- `m1spread`：kr 分 4 段读（`krv.slice`），每段后面一组 copy，读和 copy 之间都隔一个 `_fence`；最后用 `_cat0` 把 4 段拼回 kr，再读 gt。

**各参数：**

- 模板 `""`：asm 里没有指令。
- 约束 `"=s,s"`：一个 SGPR 输出、一个 SGPR 输入。
- `[x]`：调用处传的是 `i_h32`（`program_id(1) % H`），一个标量，本来就在 SGPR 里。输入是标量，整个调用只生成一段 asm，也不占 VGPR。
- `is_pure=False`：asm 有副作用。返回值没人用，但有副作用的 asm 不会被删掉。
- `dtype=gl.int32`、`pack=1`：一个 int32 输出，每段 asm 处理一个元素。

**LLIR**（`m1late`，mem1 p1，gt 的最后一条读和第一组 copy 之间）：

```llvm
%1095 = load <4 x float>, ptr addrspace(3) %359, align 16, !dbg !95, !alias.scope !75, !noalias !49
%1096 = tail call i32 asm sideeffect "", "=s,s"(i32 %.decomposed) #5, !dbg !96
; ... big 转 bf16 的 fptrunc
tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %1125, ptr addrspace(3) %119, i32 16, i32 %120, i32 0, i32 0, i32 0), !dbg !107, !alias.scope !49

attributes #5 = { nounwind }
!49 = !{!50}
!50 = !{!"amdg.AsyncCopies", !51, !"Scope containing all AsyncCopyGlobalToLocal and BufferLoadToLocal ops"}
!75 = !{!76}
!76 = !{!"amdg.LocalLoads", !51, !"Scope containing all LocalLoad ops"}
```

- `asm sideeffect` 来自 `is_pure=False`。`#5` 只有 `nounwind`，没有 `memory(...)`，所以 LLVM 认为这个调用可能读写任何内存。
- `%.decomposed` 就是 `i_h32` 的值。
- LDS 读和 copy 的 alias scope 互不重叠，它们之间没有依赖；`_fence` 和两边都有依赖。

**machine scheduler 之前的 MIR**（省略了 `implicit $exec` 和内存操作数）：

```text
%2179:vreg_128_align2 = DS_READ_B128_gfx9 %2157, 448, 0                ; gt 的最后一条读
INLINEASM &"", sideeffect attdialect, regdef:SReg_32, def dead %1240, reguse:SReg_32, %3830
undef %2284.sub0:vreg_128_align2 = V_CVT_PK_BF16_F32_e64 0, %2103.sub0, 0, %2103.sub1, 0, 0
...                                                                     ; 共 8 条 cvt
$m0 = COPY %20
BUFFER_LOAD_DWORDX4_LDS_OFFEN %714, %841, 0, 0, 0, 0, 1                 ; 第一组 copy（kd）
```

- `sideeffect` 让这条 `INLINEASM` 的 `hasUnmodeledSideEffects()` 为真。输出是 `def dead`，没人用。
- 这时的顺序就是源码顺序（正常编译在这一步也是先读后 copy）；`_fence` 起作用是在接下来的 machine scheduler 里。

**ISA：**

```text
	ds_read_b128 v[204:207], v16 offset:448
	;;#ASMSTART
	;;#ASMEND
	s_add_u32 s0, s51, 0x8000
	s_addc_u32 s1, s52, 0
	s_mov_b32 m0, s28
	s_and_b32 s1, s1, 0xffff
	buffer_load_dwordx4 v143, s[0:3], 0 offen lds
```

- asm 里没有指令，ISA 里只剩 `;;#ASMSTART`、`;;#ASMEND` 两行注释，标出 `_fence` 的位置。`m1late` 的 ISA 里有 5 段这样的空 asm（循环里 4 段，循环外 1 段），`m1spread` 有 40 段。
- VGPR 仍是 225，和正常编译一样。

**为什么挡得住访存、挡不住 ALU：** machine scheduler 建依赖图（`ScheduleDAGInstrs::buildSchedGraph`）时，有 unmodeled side effect 的指令被当作 barrier chain：它前面的每条访存都连一条边到它，它再连一条边到后面的每条访存。所以 LDS 读、copy 都越不过它。ALU 指令不访存，和它之间没有边，可以随意移动。IR 层也一样：没有 `memory(...)` 的调用被当作可能读写任何内存，IR pass 不会把 load、store 移过它。

ALU 照样会移动，`m1spread` 里就看得到：源码里 big 转 bf16 的 8 条 cvt 写在所有 `_fence` 之后，ISA 里却有 3 条越过了几个 `_fence`，跑进第 2～4 组 copy（`ssssbg`）：

```text
区域 5（mem1 p1）
  ROCm 正常编译   ssssgssssgssssgs.s.ss.sbgrrrrrrrrrrrrrrrr.rrrrrrrrbbbbbbbs
  m1late          .rrrrrrrrrrrrrrrrrrrrrrrrssssgssssbgssssbgs.s.ss.sbgbbbbbs
  m1spread        rrrrssssgrrrrssssbgrrrrssssbgrrrrss.s.s.sbg.rrrrrrrrbbbbbs
```

**和另外两种挡板比较：**

| | 生成的指令 | 访存能否越过 | ALU 能否越过 | 用在哪里 |
| --- | --- | --- | --- | --- |
| `_fence(x)` | 无 | 不能 | 能 | Triton 3.7 源码变体的 `m1late`、`m1spread` |
| `_s_barrier(x)` | 一条 `s_barrier` | 不能 | 能 | 循环外的 workgroup 同步 |
| `gl.amd.hint.sched_barrier()` | 无（MIR 里是 `SCHED_BARRIER 0`） | 不能 | 不能 | 研究 kernel 的 `_sb` |

`_fence` 的效果接近一个放行 ALU、不放行访存的 `sched_barrier`。LLVM 的 `sched_barrier` 可以用 mask 这样设置，上游 Gluon 的 `hint.py` 目前只接受 mask 0。

**结果**（ROCm Triton 3.7，T = 16384，GPU 3，和正常编译交替）：`m1late` −0.3 µs，`m1spread` −0.2 µs；和 `d2pk16w` 一起用没有额外收益。

### 5.4 后一半：copy 分散到读之间

- `M1=4`：kr 在 LDS 里按 K 切成 4 段 memdesc（`.slice`），每段 4 条读后面跟一组 copy，各占一段。ISA 已经接近 MIR 第 3 步，但比 `g_lt` 慢 0.95 µs，原因没有细查。
- 这样切的 kr 带 (16,1,8) swizzle，沿 K 切会跨 swizzle 的相位。上游的 `MemDescSubsliceOp` 只要求偏移是切片大小的整数倍，lowering 把偏移当作逻辑坐标，再经过完整的 swizzle 映射，所以地址是对的（误差不变）。
- `m1spread`（5.3 节）是同样的顺序，在 ROCm Triton 3.7 上快约 0.2 µs，比 `m1late` 少。
- 这一半值多少，可以从 `g_lt` 的 Y_1 槽估计（7.5 节）：`M1=3` 的 copy 排在读后面，copy 的 SALU 用不上读之间的空闲发射周期。

## 6. 回边 SALU 和 β clamp（第 5、8 步）

### 6.1 回边 SALU（第 5 步，`LT=1`）

**现象：** 每次迭代的最后一个 `s_barrier` 之后、跳转之前有 12 条 SALU：5 对 64 位指针递增（`s_add_u32`/`s_addc_u32`）、一条循环计数 `s_add_i32` 和一条比较 `s_cmp_lg_u32`。它们落在下一次迭代 dot1 p0 的开头，两组 wave 都要先发完它们。ROCm 正常编译的 ATT 里，dot1 p0 的第一条 VALU 在 barrier 释放后 72 个周期才发出，第一条 MFMA 在 162 个周期（MIR 第 5 步之后是 20 和 56）。上游编译的原版一样是这 12 条。

**成因：** 源码里每个 chunk 用 `ws0 + j + 3`、`tok_base + (j + 3) * C` 重新算 6 个 copy 的源地址，LSR 把它们变成 5 对 64 位的指针 IV，递增放在循环比较的前面。warp-pipeline 又把循环末尾的 border（`s_waitcnt lgkmcnt(0)` + `s_barrier`）放在循环体的最后、循环比较之前，于是这些递增都落在了 border 之后。

**MIR 第 5 步**（`--latch-early`）：把这 12 条挪到 mem2 p3 最后那串 `s_waitcnt lgkmcnt(0)` + `s_barrier` 之前，藏在等 LDS 的时间里。这些 SGPR 只被下一次迭代的 copy 和跳转读，中间只隔着不改 SCC 的 `s_waitcnt` 和 `s_barrier`，所以这样挪是合法的。

**`LT=1`：** 不关 LSR，而是在源码里直接写出这些 IV。6 个 copy 基址和 β 的上限 `T − 1 − tok`（int32）放进一个 tuple `pk`，作为循环携带的值。位置 PO（0～3，编译期常数）用的是 `pk` 加 PO 个 chunk 的偏移，这个偏移会折进 SALU 的立即数；mem2 p3 的最后再把 `pk` 推进 4 个 chunk。这样递增就在最后一个 `s_barrier` 之前：

```python
@gluon.jit
def _advance(pk, n: gl.constexpr, C: gl.constexpr, K: gl.constexpr, V: gl.constexpr,
             H: gl.constexpr):
    """pk (the six _bases and the beta limit) moved on by n chunks."""
    return (pk[0] + n * (C * K), pk[1] + n * (2 * C * C), pk[2] + n * (C * H * V),
            pk[3] + n * (C * H), pk[4] + n * (C * K), pk[5] + n * K, pk[6] - n * C)

# 循环外：chunk 3 的基址
b6 = _bases(ws_kd, ws_inv_mqk, v_input, beta_in, ws_kr, ws_gt, ws0 + 3, tok_base + 3 * C,
            C, K, V, H)
pk = (b6[0], b6[1], b6[2], b6[3], b6[4], b6[5],
      (T_TOTAL - 1 - (tok_base + 3 * C)).to(gl.int32))

# _step_g 里，位置 PO 的 mem1、mem2：本位置要发的 chunk j+3 的基址
pd = _advance(pk, PO, C, K, V, H)
# mem2 的最后（只有 PO == 3 时 ADV = 4）
if LT and ADV > 0:
    pk = _advance(pk, ADV, C, K, V, H)
```

aiter 里 `_advance` 和 `pk` 是同样的写法，但 `pk` 只有 6 个 copy 基址，没有 β 上限（3.2 节）。

**结果：** 回边从 12 条降到 4 条。`g_d2b2m3` → `g_lt`：GPU 2 49.50 → 48.35，GPU 3 48.18 → 47.06。ATT 里降的是 p0（4 个槽合计 2516 → 2290），p1～p3 反而略涨（X、Y、Z、W 合计 1839 → 1874）。

剩下的 4 条：

```text
; g_lt，区域 16（回边）
s_addk_i32   s18, 0xff80        ; β 上限 −4C
s_add_u32    s2, s2, 0x8000     ; kd、kr 合用的偏移 IV，+4 个 chunk
s_addc_u32   s3, s3, 0
s_cmp_lg_u32 s2, 0x38000        ; 循环计数也换成了这个 IV
```

- 这是 LSR 自己的选择：kd、kr 每 chunk 的步长相同（C·K 个 bf16），LSR 把两者合成一个偏移 IV，循环计数也换成了它。dot1 p0 开头因此多了 9 条 SALU（`MsssMsssMsss`），用这个 IV 算出 kd、kr 的基址和 β 上限。
- β 上限一开始写成 int64，回边是 5 条（64 位加法要 2 条），只快约 0.5 µs；改成 int32 后是 4 条，快约 1 µs（不同批次）。
- aiter 的回边是 6 条 SALU 加跳转：两个 64 位指针递增、循环计数和比较。循环计数是运行时的值，因为每个 segment 的 chunk 数不同。

**关掉 LSR 不行：** 再加 `llvm_cl_opts="disable-lsr"`（`g_ltl`），回边剩 2 条，dot1 p0 的 9 条 SALU 也没了，但 mem2 p3 的 SALU 更多，整体慢 0.4 µs。不开 `LT`、只关 LSR 也慢约 0.4 µs；ROCm Triton 3.7 上关 LSR（`--src nolsr`）也一样更慢（49.9 µs）。

`LT` 只改源码的写法，不依赖 `sched_barrier`。

### 6.2 β clamp（第 8 步，`BD1=2`）

**MIR 第 8 步：** 每个 mem1 有 4 条指令算 β copy 的源地址：`s_add_i32`（token 上界加偏移）、`v_min_i32`（clamp）、`v_mul_lo_u32`、`v_add_lshl_u32`。

- Z 槽里 sl1 的 mem1 和 sl0 的 dot2 共用一个 SIMD。dot2 开头的 32 条 pk、A 段的 cvt 都要发 VALU，mem1 的这几条只能挤在 B 段那 9 条 MFMA 之间。
- `v_mul_lo_u32` 是 1/4 速率的指令，一条占好几个发射周期，会把 dot2 的 MFMA 往后顶，β copy 也跟着晚发。
- 这 4 条的输入在 dot1 里就都有了。MIR 把它们挪到同一位置 dot1 的末尾，填进一个 `s_nop 4` 的空等。

**`BD1=2`：** β copy 的偏移在 dot1 的 stage 里算，结果 `boff` 传给 mem1 的 β copy：

```python
@gluon.jit
def _beta_off(beta_o, blim, i_h, H: gl.constexpr):
    # The upper half of the 2C beta window runs past T for the last chunk: blim = T - 1 - tok.
    return gl.minimum(beta_o, blim.to(gl.int32)) * H + i_h

# _dot1，D1=0：A 链的 4 条 MFMA 之后
if BD1 != 0:
    boff = _beta_off(beta_o, blim, i_h32, H)
```

**结果：** dot1 开头多了 1 条 SALU（β 上限），末尾多了 2 条 VALU（`v_min_i32`、`v_mul_lo_u32`，指令串里的 `III..I`）。β clamp 的第 3 条 `v_add_lshl_u32` 被 LLVM 和 copy 里的"乘 4 字节"合成了一条，留在 mem1。`g_d2m3` → `g_d2b2m3`：ATT 每 chunk 2060 → 2008（Z 槽 529 → 464）；时间上 GPU 2 快 0.58 µs，GPU 3 慢 0.25 µs，在噪声内。

aiter 的 kernel 从 K1 的 workspace 读 β，只有最后一个 chunk 的 copy 加 mask（3.2 节）。它没有 β clamp，这一步不适用。

## 7. 没有落地的几步（第 1、2、4、9、10 步）

下面的判断都以上游 Triton main 为准。

### 7.1 第 1、9 步：按单条 MFMA 交错（能写，但更慢）

**MIR 的做法：**

- 第 1 步：dot1 的 20 条 MFMA 顺序不变，用发射模型做 list scheduling，把就绪的 VALU 按紧急程度填进 MFMA 的阴影，排成一条 MFMA 隔两条 VALU（`MbbMbb…`）。同时把 dot1 的 pk 拆成 `v_mul_f32`，β 取负用的 `v_xor` 并进乘法的 neg 修饰位。
- 第 9 步：dot2 B 段的每条 MFMA 后面补发 2 条 `v_mul_f32`（`Mmm`），每两条 B MFMA 后面的 4 条凑成一个 A tile 的 srcC。

**按 piece 排的写法都更慢：**

```text
区域 4（dot1 p1）
  g_d2b2m3  MsMMnbbbMbMMnbbbMbnbbbbMmmphMhnphMMMbbbMbMnMMnmmpppphhhhhIII..Iws
  MIR       ffffffMffMbbMbbMbbMbbMbbMbbMbbMbbMmmMmmMhhMmmMmmMs...hhMmmImmIhhmmmmhhnIIws
```

- `D1=1/2` 按 K 步切段，一个 piece 是 2 条 MFMA 加 4 条 cvt，最多排成 `MMbbbbMM…`，比 `g_lt` 慢 1.12 / 0.69 µs。
- `D2S=3` 只能排成 `MM........`，慢 0.90 µs。

**按单条 MFMA 写：** `sched_barrier` 只能切段，段里的顺序由 machine scheduler 决定。每条 MFMA、每组 VALU 各写成一个 op、各占一段，顺序就钉死了。上游 Gluon 的 `gl.amd.hint.sched_barrier()` 只接受 `allow=None`（mask 0），也没有暴露 LLVM 里能按"几条 MFMA、几条 VALU、几条 DS"交替排的 `llvm.amdgcn.sched.group.barrier(mask, size, syncID)`。有了它们可以少写很多段，但不是必需的（`sched_barrier` 的非 0 mask，C++ 的 builder 已经接受，只是 `hint.py` 还没放开）。要解决的是下面两件事，都不花指令。

**一、两个 16 行的 tile 拼回 32 行。** K2 里每个 MFMA 的结果都是下一个 MFMA 的 B 操作数，而结果的行正好是下一个 MFMA 的 K。16x16x32 的 MFMA 一条只出 16 行，下一条却要 32 行的 K，所以拆到单条以后，要把两条的结果拼成一个 tensor：

- tmp → big：tmp 的行是 C（32），big = inv @ u 的 K 也是 C。dot1 的一个 K 步是 C 的两个 16 行 tile 各一条 MFMA；拆成单条，就得把 tmp 按 C 拆成上下两个 tensor，到 u 这里再拼回去。big 进 dot2 也一样，dot2 的 K 也是 C。
- dot2 → 下一个 dot1：g 的行是 K（128），下一个 dot1 按 K 每 32 行一条 MFMA。dot2 的一个 piece 是两个输出 tile（例如 B0、B1），正好是下一个 dot1 一个 K 步的操作数；拆成单条，每条只出 16 行，又要两个拼成一个 K 步。
- 沿 K 拆不用拼：研究 kernel 把 dot1 的 K = 128 拆成 4 步，每步累加到同一个 tmp 上，结果还是一个 tensor。

在 ISA 里拼不花指令：两个 tile 的 cvt 直接写进同一组 4 个 VGPR，下游 MFMA 把这 4 个当一个操作数读。

```text
; ROCm 正常编译的 MIR，dot1 p1（区域 4）：big_b 的 B 操作数 %2099 由 tmp_b 的两个 C tile 拼成
undef %2099.sub0 = V_CVT_PK_F16_F32_e64 0, %2068.sub0, 0, %2068.sub1, 0, 0   ; %2068、%2065：tile 0（%1950）乘 −β
%2099.sub1 = V_CVT_PK_F16_F32_e64 0, %2065.sub0, 0, %2065.sub1, 0, 0
%2099.sub2 = V_CVT_PK_F16_F32_e64 0, %2088.sub0, 0, %2088.sub1, 0, 0         ; %2088、%2085：tile 1（%1965）乘 −β
%2099.sub3 = V_CVT_PK_F16_F32_e64 0, %2085.sub0, 0, %2085.sub1, 0, 0
%2100 = V_MFMA_F32_16X16X32_F16 %1910, %2099, 0, 0, 0, 0                      ; big_b 的两个输出 tile（inv 的上下两半），
%2106 = V_MFMA_F32_16X16X32_F16 %1911, %2099, 0, 0, 0, 0                      ;   B 操作数都是整个 %2099

; dot2 p1（区域 6）：下一个 dot1 第 0 个 K 步的 B 操作数 %2594 由 B0、B1 拼成
undef %2594.sub0 = V_CVT_PK_BF16_F32_e64 0, %2291.sub0, 0, %2291.sub1, 0, 0   ; %2291 是 B0
%2594.sub1 = V_CVT_PK_BF16_F32_e64 0, %2291.sub2, 0, %2291.sub3, 0, 0
%2594.sub2 = V_CVT_PK_BF16_F32_e64 0, %2305.sub0, 0, %2305.sub1, 0, 0         ; %2305 是 B1
%2594.sub3 = V_CVT_PK_BF16_F32_e64 0, %2305.sub2, 0, %2305.sub3, 0, 0
; 区域 8（dot1 p2）：%2607 = V_MFMA_F32_16X16X32_BF16 %2563, %2594, %2606，C 的另一个 tile 也读 %2594
```

Gluon 没有专门的 concat，但 `gl.join` + `permute` + `reshape` 就是沿第 0 维的拼接（Triton 3.7 源码变体的 `m1spread` 也这样把 4 段 kr 拼回去）：

```python
@gluon.jit
def _cat0(a, b):
    # [M, N], [M, N] -> [2M, N]; the joined dimension stays in registers.
    return gl.reshape(gl.permute(gl.join(a, b), [2, 0, 1]), [2 * a.shape[0], a.shape[1]])


@gluon.jit
def _mt(a, b, c, t: gl.constexpr, H: gl.constexpr):
    """One MFMA, tile t (H rows) of a @ b + c, in its own sched region."""
    d = gl.amd.cdna4.mfma(_rows(a, t, H), b, c)
    _sb(True)
    return d


@gluon.jit
def _mfma2(a, b, c, H: gl.constexpr):
    """a @ b + c for a 2-tile piece as two _mt, concatenated again."""
    t0 = _mt(a, b, _rows(c, 0, H), 0, H)
    t1 = _mt(a, b, _rows(c, 1, H), 1, H)
    return gl.convert_layout(_cat0(t0, t1), c.type.layout)
```

- `D2=3`（模式 `g_d2c`）在 `D2=1` 的基础上，把 4 个 B piece 的 MFMA 都换成 `_mfma2`：每个 piece 拆成两条单条 MFMA，各占一段，再拼回 32 行。循环每 chunk 235 条指令、VGPR 229、误差 (5.43e-04, 1.88e-03)，都和 `D2=1` 相同，各类指令的条数也相同，没有 `v_mov`。dot2 的指令串一字不差，17 个区域里只有 dot1 p1～p3 各有一条 MFMA 挪了一格。
- `join` 出来的那一维留在寄存器里，拼完的 tensor 用的就是两个 tile 原来的寄存器，`convert_layout` 不生成指令。
- 研究记录最初说 Gluon 没有 concat、op 拆不到单条 MFMA，这个说法不对（研究记录 4.14.15）。

**二、一个 tile 对半拆。** MIR 每条 MFMA 后面的 2 条 `v_mul` 是一个 16 行 tile 的一半，`gl.amd.slice` 切不到（要和 16 行的 CTA tile 对齐）。改用 reshape、permute、split：

```python
@gluon.jit
def _halves(x):
    # A lane holds rows 4a + 0..3 of a 16-row MFMA tile in its 4 registers.  Splitting row bit 1
    # gives two halves of one register pair each (a pk, or two v_mul_f32 in an MFMA shadow);
    # splitting bit 0 would need v_mov to form the pairs.
    y = gl.reshape(x, [x.shape[0] // 4, 2, 2, x.shape[1]])
    return gl.split(gl.permute(y, [0, 2, 3, 1]))


@gluon.jit
def _unhalves(lo, hi, L: gl.constexpr):
    y = gl.permute(gl.join(lo, hi), [0, 3, 1, 2])
    return gl.convert_layout(gl.reshape(y, [4 * y.shape[0], y.shape[3]]), L)
```

- 16x16 MFMA 的结果，lane l 的 4 个寄存器放第 l mod 16 列的 4 行：4a + 0～3，a = l / 16。寄存器号的 bit 0 对应行号 +1，bit 1 对应 +2。
- 按行号的 bit 1 拆，每半是一对相邻的寄存器，正好是一条 pk 的操作数。pk 落在 MFMA 阴影里时，`SIPreEmitPeephole` 把它拆成 2 条 `v_mul_f32`，就是 MIR 的 `Mmm`。按 bit 0 拆，两个寄存器不相邻，要 `v_mov` 凑对。
- `_mulh(x, s, h)` 用它算 `x * s[:, None]` 的第 h 半。

**dot2，`D2=4`（第 9 步）：** 照 MIR 的 dot2 一条一条写，`D2N` 就是 MIR 的 N。先算 B 的全部 `g·gt` 和前 8 − N 个 A tile 的；B_0～B_7 各一段，前 2N 段在 MFMA 后面再算后 N 个 A tile 的半个 `g·gt`（`_mulh`）；最后 A_k 各一段，后面跟 B_k 的 bf16 cvt，每两个 tile 拼成下一个 dot1 的一个 K 步。代码是研究 kernel 的 `_dot2`。

```text
区域 6（dot2 p1）
  g_t0 (D2N=0)  ppppppppppppppppppppppppppppppppMMMMMMMMMbbMbbMbbMbbMbbMbbMbbMbbws
  MIR b1n2      wppppppppppppppppppppppppppppppppMMMMMMMMMbbMbbMbbMbbMbbMbbMbbMbbws
  g_t3 (D2N=3)  ppppppppppppppppppppppppppMmmMmmMmmMmmMmmMmmMMMbbMbbMbbMbbMbbMbbMbbMbbws
  MIR b1n2s3    wppppppppppppppppppppppppppMmmMmmMmmMmmMmmMmmMMMbbMbbMbbMbbMbbMbbMbbMbbws
  g_t4 (D2N=4)  ppppppppppppppppppppppppMmmMmmMmmMmmMmmMmmMmmMmmMbbMbbMbbMbbMbbMbbMbbMbbws
  MIR b1n2s4    wppppppppppppppppppppppppMmmMmmMmmMmmMmmMmmMmmMmmMbbMbbMbbMbbMbbMbbMbbMbbws
```

- 每一对只差开头那条 `s_waitcnt`：ROCm 编译在 dot2 开头等 mem1 的 LDS 读，上游在 mem1 末尾、`s_barrier` 之前等。
- VGPR：`g_t3`、`g_t4` 是 221，`g_t0` 和 `g_lt` 一样是 229（MIR 这几步是 207）。

**dot1，`D1=3`（第 1 步）：** 照 MIR 第 1 步写，20 条 MFMA 的先后不变（C 的 tile 0 的 B 链、tile 1 的 B 链、两条 A 链、big_b、big_a）。除了第一条，每条 MFMA 后面跟 2 条 VALU：g_a 一个 tile 的 bf16 cvt、tmp 乘 −β 的半个 tile，或者它的 f16 cvt。代码是研究 kernel 的 `_dot1`。

```text
区域 4（dot1 p1）
  g_lt        MsMMnbbbMbMMnbbbMbMbbbbnppMmmhhMhphMMbbbMbMnMMInmmppphhhhInI..Iws
  g_d1t       MsMbbMbbMbbMbbMbbMbbMbbMbbMmmMmmMnhhMmmMnmmM..hhMmmImmInmmphhhhnIIws
  MIR cyc03   ffffffMffMbbMbbMbbMbbMbbMbbMnbbMbbMmmMmmMhhMmmMmmMhhMmmImmIhhmmmmhhnIIws
  MIR b1n2    ffffffMffMbbMbbMbbMbbMbbMbbMbbMbbMmmMmmMhhMmmMmmMs..n.hhMmmImmIhhmmmmhhnIIws
```

- MIR 开头的 `ffffffMff` 是 v 转 f32，ROCm 编译放在 dot1 开头，上游放在 mem2。`g_d1t` 里 A_6 后面的 `..` 是 β clamp（`BD1=2`），对应 MIR 第 8 步（b1n2）的 `s..n.`。
- 和 MIR 的差别：A_3、A_5 后面各多一个 `s_nop 4`，是寄存器分配把 MFMA 刚读完的 srcC 寄存器给了紧跟着的 VALU，和第 10 步用 tie 去掉的是同一类（7.4 节）；末尾有 1 条 pk 没拆（`mmp`，MIR 是 `mmmm`）。

**指令数：** 整个循环（4 个 chunk），都在 `g_lt` 的开关上再改 dot1、dot2。所有模式的误差都是 (5.43e-04, 1.88e-03)，没有 spill。

| 模式 | 开关 | VGPR | 指令 | `s_nop` |
| --- | --- | --- | --- | --- |
| `g_lt` | `D2=1` | 229 | 933 | 21 |
| `g_t0` | `D2=4` | 229 | 932 | 20 |
| `g_t3` | `D2=4`、`D2N=3` | 221 | 957 | 21 |
| `g_t4` | `D2=4`、`D2N=4` | 221 | 962 | 18 |
| `g_d1t` | `D1=3`、`D2=1` | 229 | 946 | 15 |
| `g_d1t3` | `D1=3`、`D2=4`、`D2N=3` | 221 | 973 | 18 |
| `g_d1t4` | `D1=3`、`D2=4`、`D2N=4` | 225 | 977 | 14 |
| MIR b1n2s4t1 | — | 207 | 948 | 8 |

**计时和 ATT：** T = 16384，13 个模式在一个进程里 20 轮交替；GPU 3、GPU 2 轮流各跑 5 次，一次只跑一个进程。表里是每次运行里相对 `g_lt` 之差的平均，方括号是 5 次里的最小、最大值。ATT 在 GPU 2 上抓，X、Y、Z、W 是 p1～p3 的平均。

| 模式 | 相对 `g_lt`，GPU 2（µs） | 相对 `g_lt`，GPU 3（µs） | ATT 每 chunk | X / Y / Z / W |
| --- | --- | --- | --- | --- |
| `g_lt` | 0（48.47） | 0（47.38） | 1976 | 447 / 505 / 476 / 445 |
| `g_t0` | +0.08 [−0.17, +0.51] | +0.09 [−0.14, +0.39] | 1975 | 437 / 503 / 471 / 431 |
| `g_t3` | +0.53 [+0.41, +0.61] | +0.36 [−0.25, +0.69] | 2037 | 437 / 503 / 560 / 424 |
| `g_t4` | +1.45 [+1.26, +1.69] | +1.05 [+0.54, +1.20] | 2111 | 441 / 499 / 652 / 409 |
| `g_d1t` | +0.77 [+0.67, +0.88] | +0.65 [+0.37, +1.34] | 2064 | 465 / 556 / 495 / 455 |
| `g_d1t3` | +0.87 [+0.80, +0.94] | +0.65 [+0.09, +0.87] | 2129 | 465 / 581 / 575 / 429 |
| `g_d1t4` | +1.60 [+1.45, +1.98] | +1.49 [+1.05, +2.15] | 2122 | 444 / 531 / 617 / 432 |

- 只把 dot2 拆成单条 MFMA（`g_t0`，`D2N=0`）和 `g_lt` 持平。
- 第 9 步多出来的周期都在 Z 槽（sl0 的 dot2 加 sl1 的 mem1），W 槽（sl0 的 mem2 加 sl1 的 dot2）反而略少。同一段 dot2 和 mem1 同槽时变慢、和 mem2 同槽时变快，原因没有细查。MIR 上第 9 步快 0.2～0.3 µs，但它是在第 1～8 步之后加的，`g_lt` 还缺第 1、2、4 步和第 3 步的后一半。
- 第 1 步多出来的主要在 Y 槽（sl1 的 dot1 加 sl0 的 mem1）。MIR 上单独做第 1 步也慢 0.54 µs（ATT 每 chunk 多 70），要和第 2 步一起才有收益，而第 2 步写不出来（7.2 节）。
- 两步一起（`g_d1t3`、`g_d1t4`）不比单独做好。
- `g_d2c` 比 `g_d2` 慢 0.31、0.42 µs。它的 ATT 每 chunk 是 2150，`g_d2` 两次是 2106、2144，差别在 `g_d2` 自己的波动之内。
- 2026-11-09 白天节点出了问题，第一次计时没做成；表里的数是节点恢复后、驱逐计数为 0 时测的（3.4 节）。

**第 1 步的另外两项：**

- β 取负的 `v_xor`：上游编译的 dot1 里本来就没有。
- dot1 的 pk 拆成 `v_mul_f32`：拆不拆由 `SIPreEmitPeephole` 按 pk 落不落在 MFMA 阴影里决定，跟着顺序走，没有单独的开关。在 ROCm Triton 3.7 上试过把整个 kernel 的 pk 都拆掉（`--src scal`，即 `AMDGCN_SCALARIZE_PACKED_FOPS=1`），VGPR 209，49.3～49.7 µs，比正常编译慢。

### 7.2 第 2 步：mem2 的地址 add

**MIR 第 2 步：** mem2 里 inv、v、β 三个读的地址要用 VALU 算（`v_add_u32`），这 3 条夹在读之间。第 1 步把 dot1 排密以后，轮到它们时同槽的 dot1 正在密集发射，add 发不出去，后面的读也跟着等：inv 地址之后的那条读等了 172 周期（正常编译是 24）。MIR 把这 3 条 add 移到区域开头，后面钉一个 `SCHED_BARRIER 0`。

**写不出来的原因：** 这几条 `v_add_u32` 是 LDS 读（`load_shared_relaxed`，现在是 `smem.load`）在 lowering 时生成的，源码里没有对应的 op，不能和读分开挪。`M2=1/2` 把 inv、v、β 的读整个挪到 kd 前面，add 跟着到了前面（`.rr.rr.rr…`），但慢 1.28 / 0.60 µs；和 `D1=2` 一起开（`g_m2d1`）也慢 0.53 µs。这一步要解决的问题本来就是第 1 步带来的（MIR 里第 1、2 步要一起才有收益），第 1 步有结果之前，它值多少也定不了。

### 7.3 第 4 步：`+0` 拷贝

**现象：** 循环里有 11 条 `v_add_u32 v, 0, v`，都在 p0：9 条在 mem1 p0（kr 的 8 个 TR 读地址，加上 gt、β 共用的基址），2 条在 mem2 p0（inv、v 的读地址）。ROCm 正常编译的 ATT 里，和 dot2 同槽运行的那组 mem1 p0 里，这 9 条每条约 12 个周期，第一条 LDS 读到约 144 个周期才发出；MIR 里删掉以后是约 12 个周期。上游编译出来一样：

```text
区域 1（mem1 p0）
  g_lt      .....rrrr.rrrr.rrrr.rrrr..rrrrrrrrssgsssgsss.gsssbgbbbbbbbws
  MIR       .rrrrssssssgrrrrssssgrrrrssssgrrrrssssgrrrrrrrrbbbbbbbbs
```

开头的 `.....` 和每 4 条读之间的 `.` 就是这 9 条，mem2 p0 还有 2 条。下面以 ROCm Triton 3.7 的编译为例，kr 的第一个 TR 读：

```text
; LLIR，循环外：LICM 已经把地址提出了循环
%296 = getelementptr inbounds nuw i8,
         ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 32768), i32 %295
; LLIR，循环里
%818 = tail call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %296)

; MIR，machine scheduler 之前，循环块
%1649 = V_ADD_U32_e32 0, %1375
undef %1776.sub0_sub1 = DS_READ_B64_TR_B16 %1649, -32768, 0 :: (load (s64) from %ir.sunkaddr281)

; ISA
v_add_u32_e32 v134, 0, v153
ds_read_b64_tr_b16 v[198:199], v134 offset:32768
```

MIR 里的 `-32768` 和 ISA 里的 `offset:32768` 是同一个 16 位立即数，MIR 按有符号数打印。

**成因：**

1. Triton 把 LDS 声明成零大小的外部全局变量（动态 LDS）：`@global_smem = external addrspace(3) global [0 x i8]`。它的地址就是 LDS 的起点 0，但在 LLVM IR 里不是常量，InstSimplify、DAGCombiner 对 `add x, 0` 的折叠都看不到这个 0。
2. CodeGenPrepare 把 LICM 提出去的地址计算 `@global_smem + %x` 又沉回循环里（`sunkaddr`）。AMDGPU 的 `isLegalAddressingMode` 认为 LDS 支持"寄存器 + 寄存器 + 立即数"，于是常数部分进了 `ds_read` 的 offset，剩下的加法留在循环里。CodeGenPrepare 在同一个基本块里对同一个地址只沉一次，展开后的 4 个 chunk 在同一个基本块里，所以这些加法都出现在第一次用到的地方，也就是 p0。
3. 指令选择把 `@global_smem` 的地址变成伪指令 `GET_GROUPSTATICSIZE`，`finalize-isel` 再把它换成 `S_MOV_B32 0`。到这一步才知道它是 0。
4. MachineLICM 不肯把这条加法提出循环：寄存器压力太高（"Can't remat / high reg-pressure"）。
5. `si-fold-operands` 把立即数 0 折进加法，得到 `V_ADD_U32_e64 0, %x`。它的 `tryConstantFoldOp` 会消掉与 0 的 or、and、xor，以及标量的 `S_ADD`/`S_SUB` 0（[llvm/llvm-project#198410](https://github.com/llvm/llvm-project/pull/198410)，2026-05 合并），但不处理 VALU 的 `V_ADD_U32 0, x`。K2 的地址每个 lane 不同，用的正是 VALU。上游 Triton 现在用的 codegen 库 08b05a33、下一个 pin 293a85c0 都是这样。
6. `si-shrink-instructions` 把它缩成 `V_ADD_U32_e32 0, %x`，一直留到 ISA。

**修复：** [llvm/llvm-project#163298](https://github.com/llvm/llvm-project/pull/163298)（"[AMDGPU] Fold dst = v_add 0, src -> src"），2025-10-14 提交，2026-10-21 获批准（等 rebase），截至 2026-11-09 还没合并。合并之后，还要等 Triton 把 AMD codegen 用的 LLVM 升到包含它的版本。

**收益估计**（都是在 ROCm Triton 3.7 的编译上做的）：

- 用 `llc` 模拟这个 PR（在 `si-fold-operands` 之后把 `V_ADD_U32 0, x` 换成 x）：循环里的 `+0` 拷贝全部消失，循环 974 → 958 条指令，VGPR 225 → 217。
- 只用 MIR 删掉它们（MIR 手改脚本的 `--algo none --fold-zero-add`）：VGPR 225 → 215，12 组交替计时全部更快，GPU 2 上 50.74 → 50.44 µs，GPU 3 上 49.45 → 49.04 µs，约快 0.3～0.4 µs。
- 在 MIR 第 1～3 步之上再删（第 4 步），累计从 −1.17 到 −1.88 µs（1.2 节的表）。

**源码里的绕法都更差：**

| 做法 | 结果 |
| --- | --- |
| 关掉 CodeGenPrepare（`llvm_cl_opts="disable-cgp"`，`g_cgp`） | 循环里的 `+0` 拷贝没了，但 VGPR 256、90 个 spill，195 µs |
| 把 LDS 基址改成有大小的全局变量（改 LLIR） | 11 条全部去掉，但所有地址都被提出循环，VGPR 255、30 个 spill |

### 7.4 第 10 步：MFMA tie

**MIR 第 10 步：** dot1 的 16 条 bf16 MFMA 串成 4 条累加链，每条 4 节，后一节把前一节的结果当 srcC 读。

- 没有 tie 时，每一节的结果写进一组新的寄存器，读完的那组 srcC 马上被寄存器分配给紧跟着的 cvt。
- 可这组寄存器刚被链上前一节写过、又正被这一节读：VALU 再写它，要和前一节隔 8 条指令（WAW）、和这一节隔 3 条（WAR）。实际只隔 3 条，编译器补一条 `s_nop 4`，后面的 MFMA 跟着晚发约 16 周期。
- 第 10 步在 MIR 里把每一节的结果改名成 srcC 的虚拟寄存器（D = C，原地累加），寄存器分配只能给两者同一组寄存器，这类 `s_nop` 就没了。

**写不出来的原因：**

- 16x16x32 的 MFMA 只有 D、C 不绑定的 opcode（`V_MFMA_…_vgprcd_e64`）。D 绑定在 C 上的 `_mac` 形式只给 D 超过 4 个 VGPR 的 MFMA（例如 32x32）定义。
- Gluon 的 `mfma(..., cd_regclass=...)` 只能选结果的寄存器类，不能让结果和 srcC 共用寄存器。
- `inline_asm_elementwise` 的输入要广播成同一个形状、同一个 layout，写不出 A、B、C 形状各不相同的 MFMA。
- 需要 MFMA 提供"结果写回累加器"的选项。

### 7.5 剩下的差距

`g_lt` 比 MIR b1n2s4t1 每 chunk 多 315 周期（ATT，GPU 2）：

- **p0 约 147：** mem1 p0 的 9 条、mem2 p0 的 2 条 `+0` 拷贝（7.3 节）；dot1 p0 开头 LSR 的 9 条 SALU，回边的 4 条 SALU（6.1 节）。
- **p1～p3 约 169：** 每个位置 X、Y、Z、W 各多 50、66、59、49。两个最典型的槽：
  - **Y_1（sl0 的 mem1 p1 加 sl1 的 dot1 p1）：** sl0 比 sl1 晚到 barrier 66 周期。16 条 kr 读约 190 周期（大约每 12 周期一条，两条之间 wave 有空闲的发射周期），copy 基址的 SALU 约 30，8 条 gt 读约 110，4 组 copy 约 100（每组 4～6 条 SALU，含写 `m0`），8 条 cvt 约 40。`M1=3` 的 copy 排在读后面，copy 的 SALU 用不上读之间的空闲；MIR 第 3 步的"4 读 1 copy"用的正是这些空闲（5.4 节）。
  - **X_2（sl0 的 dot1 p2 加 sl1 的 mem2 p2）：** sl0 比 sl1 晚到 104 周期。dot1 的尾巴是一条依赖链：最后一条 bf16 MFMA 在 312 周期发出，接着 `s_nop 4`，348～372 乘 −β，376～396 转 f16，400～448 才发出 4 条 f16 MFMA。MIR 第 1 步的重排、第 10 步去掉寄存器分配造成的 `s_nop`，缩短的就是这一段（7.1、7.4 节）。

## 8. 参考

- Triton：
  - 0df169311f：aiter kernel 测试用的上游 main（2026-11-12）。
  - f8cddb947b：第 2～6 节测量用的上游 main。173580104f：2026-11-08 的 main，7.1 节和复测用。
  - [triton-lang/triton#11932](https://github.com/triton-lang/triton/pull/11932)：`gl.amd.hint.sched_barrier`（2026-09-24）。
  - [triton-lang/triton#12345](https://github.com/triton-lang/triton/pull/12345)：buffer descriptor 和基于 descriptor 的 buffer 接口（2026-10-25）。
  - [triton-lang/triton#12478](https://github.com/triton-lang/triton/pull/12478)：废弃 `load_shared_relaxed`（2026-10-26）。
  - [triton-lang/triton#12527](https://github.com/triton-lang/triton/pull/12527)：废弃旧的 buffer 接口（2026-10-27）。
  - [triton-lang/triton#12496](https://github.com/triton-lang/triton/pull/12496)：AMD codegen LLVM 升到 08b05a33（2026-10-27）。
  - [triton-lang/triton#12656](https://github.com/triton-lang/triton/pull/12656)：从源码构建 AMD codegen LLVM 的 pin 改成 293a85c0；下载的预编译库（`cmake/amd-llvm-info.json`）仍是 08b05a33。
  - [triton-lang/triton#12664](https://github.com/triton-lang/triton/pull/12664)：`WarpSpecializeUtility.cpp` 里的窄化，用 GCC 13 构建要加 `-Wno-error=narrowing`。
  - `python/triton/experimental/gluon/language/amd/`：`hint.sched_barrier`、`slice`、`cdna4.mfma`（`cd_regclass`）、`cdna4.buffer`。
- LLVM：
  - [llvm/llvm-project#163298](https://github.com/llvm/llvm-project/pull/163298)：[AMDGPU] Fold dst = v_add 0, src -> src（未合并）。
  - [llvm/llvm-project#198410](https://github.com/llvm/llvm-project/pull/198410)：`SIFoldOperands` 折叠标量的 `S_ADD`/`S_SUB` 0（已合并）。
  - `llvm/lib/Target/AMDGPU/SIFoldOperands.cpp`：`tryConstantFoldOp`。
  - `llvm/lib/Target/AMDGPU/SIISelLowering.cpp`：`LowerGlobalAddress`、`GET_GROUPSTATICSIZE`、`isLegalAddressingMode`。
  - `llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp`：`getCountedWaitStates`、`checkMAIVALUHazards`，以及 inline asm 的 dst forwarding 规则。
  - `llvm/lib/CodeGen/ScheduleDAGInstrs.cpp`：`buildSchedGraph` 里 barrier chain 的处理。
- 研究 kernel、MIR 手改脚本和完整的实验记录不在这个仓库里。实验记录是 `k2_ideal_schedule_budget.md` 第 4.14 节：4.14.12 是 ROCm Triton 3.7 上的源码落地，4.14.13 是 MIR 逐步对照，4.14.14 是上游 Triton 上按 piece 的写法，4.14.15 是按单条 MFMA 的写法和最新 main 的复测。
