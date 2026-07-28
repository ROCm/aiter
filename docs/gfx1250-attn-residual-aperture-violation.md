# attn-residual fused kernel:越界访存(aperture violation)分析

**结论(TL;DR)**:`_attn_res_fused_kernel` 稳定触发 `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`。触发开关已收敛到 H 循环的软件流水线 `tl.range(..., num_stages=2)`,但**问题不在 Triton 的 pipeliner** —— TTGIR 层面的 predication 经逐条验算是完全正确的(§5.2)。真正的可疑点在 **AMD backend 把 `tt.load` 降级成 `buffer_load` 后 mask 的实现方式**:mask 不做 exec 收窄,而是把无效 lane 的 voffset 换成哨兵 `0x80000000`,**访存指令照常发射**,靠 SRD 的 `num_records` 让硬件丢弃(§6)。这条路径的正确性押在几个 GFX12 上存疑的 SRD 配置假设上,而 `num_stages` 只是改变了 SRD 的构造方式与哨兵 lane 的数量。

对照矩阵(其余条件全部相同):

| 配置 | 结果 |
|---|---|
| `num_stages=2`,寻址越界 | 开跑 23 秒 ~ 2 分钟内 fault |
| `num_stages=2`,寻址已 clamp 到全部界内 | **仍然 fault**(现场 `T=1066`) |
| `num_stages=1`,寻址已 clamp | **全量 GSM8K 1319 题跑通,47 分钟 0 fault,精度 0.9598 vs 基线 0.9613** |
| `num_stages=2` + `AMDGCN_USE_BUFFER_OPS=0` | 编译产物已确认改走 `global_load` + exec mask(§6.4);**端到端待验证** |

- 出问题的 kernel:`_attn_res_fused_kernel`(`atom/models/kimi_k3_fused.py`);同样寻址模式的还有 `_attn_res_reduce_kernel`、`_attn_res_combine_kernel`
- **本文自包含**:出问题的 kernel、IR dump 脚本、op 级 reproducer 的**完整源码都内联在 §8**,无需查阅其他文件。环境版本见 §1。
- 引用到的 IR/汇编片段都直接贴在 §5.2、§6.1~§6.3;原始产物(`ns1.ttgir` `ns1.amdgcn` `ns2.ttgir` `ns2.amdgcn` `ns2_nobuf.amdgcn`)可另行索取,也可用 §8.2 的脚本几秒钟重新生成。

---

## 1. 环境与版本

### 1.1 Triton(重点)

| 项 | 值 | 取得方式 |
|---|---|---|
| Triton 版本 | `3.8.0+git5b5a3760` | `importlib.metadata.version("triton")` |
| Triton commit | `5b5a3760b` "Generate new internal llvm build" | `git -C /app/triton-mi450 log --oneline -1` |
| `git describe` | `llvm-build-92f116e9` | `git -C /app/triton-mi450 describe --tags --always` |
| 前两个 commit | `186861568` "Enable Tests for Partition Conflicts Resolution for MXFP GEMM (#107)"、`c75a012f3` "Add missing tdm wait (#115)" | 同上 `-3` |
| 源码路径(容器内) | `/app/triton-mi450` | — |
| 后端 | AMD (`third_party/amd`),`HIPBackend` | — |
| `AMDGCN_USE_BUFFER_OPS` | 默认 **`True`** | `triton.knobs.amd.use_buffer_ops` |

**注意两点**:

1. 该构建**没有** `triton.__version__` 属性(`AttributeError`),必须用 `importlib.metadata.version("triton")` 取版本。
2. 这是一个**内部 fork**(commit message 里的 PR 号 `#107`/`#115` 属于内部仓库编号,不对应上游 triton-lang/triton 的 PR),`git describe` 给出的 `llvm-build-92f116e9` 是内部 LLVM 构建标记而非 Triton release tag。定位时请以 `/app/triton-mi450` 的实际源码为准,上游同名文件可能已经不同。

### 1.2 编译器 / 运行时 / 驱动

| 项 | 值 |
|---|---|
| ROCm | `7.14.0` |
| HIP | `7.14.60850-0000000`(`torch.version.hip = 7.14.60850`) |
| AMD clang | `23.0.0git`,`ROCm/llvm-project` `46fcb339fb61119b337f973c7ca9e710a319fdd0` `+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6` |
| PyTorch | `2.11.0+rocm7.14.0a20260623` |
| Python | `3.12.3` |
| amdgpu 驱动 | `6.19.0.31300009` |
| 宿主内核 | `6.8.0-38-generic` |
| 容器镜像 | `rocm/fw-bringup:gfx1250-atom-dev-20260715-tp4_pro_flash`(`sha256:78504c9b375b032b57e80013900379e9d01e7adca2dcd1b213bf0cedd3bbd721`) |

### 1.3 硬件

| 项 | 值 |
|---|---|
| arch | **`gfx1250`** |
| 设备名 | AMD Radeon Graphics |
| CU 数 | 256 |
| **wave size** | **32**(wave32;故 `num_warps=4` = 128 线程) |
| 单卡显存 | 432 GiB |
| 卡数 / 并行 | 4 / TP4 |

### 1.4 崩溃现场参数

```
T=1066  B=3  H=7168  BP=4  BLOCK_H=1024
num_stages=2  num_warps=4  DO_ADD=True  WRITE_PREF=True
dtype=bf16
```

`H = 7168` 能被 `BLOCK_H = 1024` 整除(7 次迭代);`BP = next_pow2(B+1) = 4` 而 `B = 3`。

## 2. 症状

Kimi-K3 推理(TP4,并发 64 跑 GSM8K)在开跑后**数十秒内**崩溃:

```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
... an illegal memory access was encountered
```

两个使排查困难的特征:

1. **报错位置误导**:HIP 异步语义下 fault 常在之后某个无关调用上才浮出。我们最早看到的堆栈落在 MoE 的 `flydsl_moe_fused_route_quant_scatter` 里的 `torch.zeros(E, ...)`(一个纯分配调用),据此排查会完全走偏。
2. **崩溃与否取决于内存布局**:同一份代码有时几十秒崩、有时能跑更久,报错层号每轮不同。

## 3. 如何坐实是这个 kernel

在**同一次 launch 的前后各排空一次队列**,把"别人脏的锅"和"自己越界"分开:

```python
if _ATTN_RES_SYNC and not torch.cuda.is_current_stream_capturing():
    torch.cuda.synchronize()          # 干净 => 此前所有 kernel 都没问题
_attn_res_fused_kernel[(T,)](...)
if _ATTN_RES_SYNC and not torch.cuda.is_current_stream_capturing():
    torch.cuda.synchronize()          # 报错 => fault 只能来自这一次 launch
```

实测:`already faulted BEFORE ... launch` 出现 **0 次**,而 launch 后的同步**每次都报错**并 dump 出操作数。两次独立运行的现场:

```
kernel ITSELF faulted (T=1166 B=3 H=7168 BP=4 BLOCK_H=1024 ns=2 nw=4 do_add=True)
  block_residual[shape=(1166, 3, 7168) stride=(21504, 7168, 1) ptr=0x7b851704b000]

kernel ITSELF faulted (T=1066 B=3 H=7168 BP=4 BLOCK_H=1024 ns=2 nw=4 do_add=True)
  block_residual[shape=(1066, 3, 7168) stride=(21504, 7168, 1) ptr=0x72f791831000]
```

注:`torch.cuda.synchronize()` 在 graph capture 期间是非法的(`hipErrorStreamCaptureUnsupported`),探针必须跳过 capture,否则会把这个限制误报成 fault。

## 4. 一处真实的越界寻址(已修,但**不是** fault 的原因)

候选轴 `Bp = B + 1` 被 pad 到 2 的幂用于向量化:

```python
Bp = B + 1
BP = triton.next_power_of_2(Bp)      # B=3 -> BP=4
```

kernel 内原先直接用 `b_idx` 寻址:

```python
b_idx = tl.arange(0, BP)             # 取到 BP-1,而 br 只有 B 行
br_base = t * stride_br_t + b_idx * stride_br_b
```

`b_idx >= B` 的 lane 为 `block_residual` 中不存在的行算出了地址。这些 lane 的值从不参与结果(load 上有 `(b_idx < B)` 的 mask;`b_idx == B` 那条 lane 的值由 `tl.where(is_last, ps, br)` 从 `prefix_sum` 取),但地址确实越界。代入现场数字,`t = T-1 = 1165`、`b_idx = 3`:

```
偏移 = 1165 * 21504 + 3 * 7168 = 25,073,664
br.numel() = 1166 * 3 * 7168   = 25,073,664      <-- 恰好是尾后第一个元素
末尾地址 = 0x7b851704b000 + 1166*3*7168*2 = 0x7b851a023000   <-- 正好 4KB 页对齐
```

即越界读打在紧随其后那一页的起始处。已把参与寻址的下标 clamp 回界内:

```python
b_safe = tl.minimum(b_idx, tl.maximum(B - 1, 0))
br_base = t * stride_br_t + b_safe * stride_br_b
```

这是纯寻址修复、不改数值(被 mask 的 lane 恒得 `other=0.0`,与地址指向哪行无关),`B = 1..8` 与 torch 参考对照的最大相对误差 `1.2e-3 ~ 3.9e-3`,处于 bf16 正常区间。

**但打上这个修复后 fault 依旧**,现场变成 `T=1066`。clamp 后 `br_base` 最大为

```
1065 * 21504 + 2 * 7168 = 22,916,096   (+ cols 最多 1023 => 22,917,119)
br.numel() = 1066 * 3 * 7168 = 22,923,264        <-- 在界内
```

所以这处越界寻址是**真实存在、值得修**的隐患,但**不是这次 aperture violation 的触发原因**。这一点很重要:它说明"上层把地址算对"并不足以避免 fault。

## 5. 两个已被排除的假设

### 5.1 所有指针都在界内(源码级验算)

以现场 `T=1066, B=3, H=7168, BLOCK_H=1024` 计,全部在界内:

| 指针 | 最大偏移 | 容量 | 结论 |
|---|---|---|---|
| `br_ptr + br_base + cols` | 22,917,119 | 22,923,264 | 界内(clamp 后) |
| `ps_ptr + t*stride_ps_t + cols` | 7,638,687 | 7,641,088 | 界内 |
| `hs_ptr + t*stride_hs_t + cols` | 7,638,687 | 7,641,088 | 界内 |
| `pref_ptr + t*stride_pref_t + cols` | 7,638,687 | 7,641,088 | 界内 |
| `y_ptr + t*stride_yt + cols` | 7,638,687 | 7,641,088 | 界内 |
| `sw_ptr + cols` | 7,167 | 7,168 | 界内 |

`H = 7168` 能被 `BLOCK_H = 1024` 整除(7 次迭代),`h_mask` 在所有真实迭代上全真,不存在尾块越界。

### 5.2 pipeliner 的 predication 是正确的(TTGIR 级验算,**证伪了此前的猜测**)

我们一度猜测 `num_stages=2` 预取了**第 8 次(不存在的)迭代**,地址 `base + H` 恰好落在 tensor 尾后。**dump 出 TTGIR 后这个猜测不成立**,`logs/tritonir/ns2.ttgir`:

```mlir
%acc_dot_35 = arith.subi %H, %c1024_i32 : i32                        // 上界 = H - BLOCK_H
%acc_dot_36:8 = scf.for %acc_dot_90 = %c0_i32 to %acc_dot_35 step %c1024_i32 ... {
  %acc_dot_97 = arith.addi %acc_dot_90, %c1024_i32                   // 下一次迭代的 h0
  %cols_101   = arith.addi %cols_99, %cols_5                         // 下一次迭代的 cols
  %h_mask_103 = arith.cmpi slt, %cols_101, %h_mask_6                 // 用下一次迭代的 cols 算 mask
  %br_107     = arith.addi %br_base, %acc_dot_97                     // 预取地址
  %br_110     = amdg.buffer_load %br_ptr[%br_109], %br_106           // 地址与 mask 严格配对
```

逐条核对:

- 循环上界是 `H - BLOCK_H = 6144`(**exclusive**),故 `h0 ∈ {0, 1024, ..., 5120}` 共 6 次,预取的 `h0 + 1024` 最大 **6144 < 7168**,全部界内;
- 最后一次迭代(`h0 = 6144`)由 pipeliner peel 出的 **epilogue** 处理,不产生预取;
- 预取的 mask `%h_mask_103` 是用**下一次迭代**的 `%cols_101` 重新算的,与预取地址 `%br_109` 严格配对,没有沿用当前迭代的 mask。

**结论:Triton 的软件流水线在 TTGIR 层面完全正确,predication 没有丢失,预取地址没有越界。** 请勿再从 pipeliner 方向排查。

## 6. 定位:AMD buffer ops 的 mask 实现

既然 TTGIR 正确,问题只能在其下游。对比三份汇编(`logs/tritonir/`):

| 产物 | 访存指令 | 哨兵 `0x80000000` | exec 收窄 | SRD 常量 `0xffffff` |
|---|---|---|---|---|
| `ns1.amdgcn` | 13 × `buffer_load_b128` | 13 | — | 无 |
| `ns2.amdgcn` | 26 × `buffer_load_b128` | 24 | — | **有**(`s46`) |
| `ns2_nobuf.amdgcn` | 26 × `global_load_b128` | **0** | **61** | — |

### 6.1 mask 不做 exec 收窄,而是发射带哨兵偏移的访存

`ns2.amdgcn`:

```asm
s_and_b32 vcc_lo, s49, s27                                  ; vcc = mask
v_cndmask_b32_e32 v63, 0x80000000, v2, vcc_lo               ; 无效 lane 的 voffset 换成哨兵
buffer_load_b128 v[22:25], v63, s[44:47], null offen        ; 访存照常发射
```

设计意图见 `third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.h:39-52`:

```
// Also note that buffer operations support out-of-boundary memory access.
// I.e., if offset[i] > mem_desc.num_records the operation is a nop for the i-th thread.
//
// This can be exploited to support masked operations:
//     mem_desc.num_records = max_int_32
//     oob_offset = max_int_32+1
//     masked_offset = (pred ? offset : oob_offset)
```

即 mask 的正确性**完全依赖硬件按 `offset >= num_records` 丢弃访存**。这与 `global_load` 路径有本质区别:后者用 exec mask,无效 lane **根本不发射访存指令**(§6.4)。

**这条路径每次 load 都会被走到**,不是边界情况:`BP = 4`、`B = 3`,故 `b_idx < B` 中 **lane 3 恒为假**,`br` 的每一次 load 都有 1/4 的 lane 带着哨兵偏移在飞。

### 6.2 疑点 A:`num_records` 与设计注释不一致,且随 `num_stages` 变化

汇编里 SRD 的 dword2(`num_records`)是一个**硬编码常量**,与 tensor 真实大小无关:

```asm
s_mov_b32 s46, 0xffffff        ; ns2.amdgcn:29
s_mov_b32 s42, s46             ; :55    -> s[40:43] 的 num_records
s_mov_b32 s6,  s46             ; :100   -> s[4:7]
s_mov_b32 s30, s46             ; :143   -> s[28:31]
```

两点存疑:

1. **数值与注释不符**:`BufferOpsEmitter.h` 的设计是 `num_records = max_int_32`(`0x7fffffff`),配 `oob_offset = 0x80000000`;实际发出的是 `0xffffff`。`0x80000000 > 0xffffff` 仍成立,故哨兵仍能被挡,但这说明 SRD 的构造与文档化的设计已经不一致,值得确认 GFX12 的 45-bit `num_records` 打包是否正确(见下)。
2. **`num_stages` 会改变 SRD 的构造**:`ns1.amdgcn` 里**完全没有** `0xffffff` 这个常量,各 SRD 的 dword2 统一来自寄存器 `s30`;`ns2.amdgcn` 才出现立即数 `0xffffff`。这是 `num_stages` 与访存正确性之间目前唯一可见的耦合点,也是我们认为最值得查的地方。

另外,`num_records` 无论取 `0xffffff` 还是 `max_int_32`,都意味着硬件放行的窗口是 `[base, base + 16MB/2GB)`。而 SRD 的 base 是 per-program 的 `br_ptr + t * stride_br_t`,对最后几行 `t` 而言 tensor 只剩几十 KB。**buffer ops 因此不提供任何真实的越界保护**,任何寻址瑕疵都会直接变成打到未映射页的真实访存 —— 这正是 §4 那处越界能造成 aperture violation 的原因。

### 6.3 疑点 B:GFX12 上 `OOB_select` 被丢弃(源码自承)

`BufferOpsEmitter.cpp:40-78` 构造 flags:

```cpp
uint32_t flags = (7 << 12) | (4 << 15);
if (llvm::is_contained({RDNA2, RDNA3, RDNA4, GFX1250}, targetInfo.getISAFamily())) {
  flags |= (1 << 24);
  uint32_t oob = 3;          // bits 28-29: Out of bounds select (RDNA only)
  flags |= (oob << 28);
}
```

同一处的注释承认这个 `oob = 3` 在 GFX12+ 上**不会生效**:

```
// For GFX12+ (RDNA4, GFX1250): LLVM's lowerPointerAsRsrcIntrin()
// (SIISelLowering.cpp) rebuilds the descriptor in v2i64 format (57-bit
// base, 45-bit num_records) and shifts the flags operand left by 28 bits
// into bits [127:124]. Therefore only flags bits [3:0] survive:
//   bit 1 -> bit 125: OOB_select (0=structured, 1=check offset only)
// OOB_select=0 is correct for raw buffer ops with stride=0
// (structured and unstructured modes are equivalent in this case).
```

`flags = 0x31027000` 的 bits[3:0] 为 0,故实际 `OOB_select = 0`(structured)。整个 mask 机制于是押在注释最后那句**假设**上:"stride=0 时 structured 与 unstructured 等价"。如果这个假设在 gfx1250 上不成立(或 45-bit `num_records` 的打包与 `0xffffff` 的写入方式不匹配),哨兵 `0x80000000` 就不会被丢弃,而是真的去访问 `base + 0x80000000` —— 距任何映射区 2GB 之外,**其后果恰好就是 aperture violation 而非读到邻页垃圾**,与观察到的错误类型吻合。

### 6.4 决定性对照:关掉 buffer ops

Triton 有开关 `AMDGCN_USE_BUFFER_OPS`(`python/triton/knobs.py:517`,默认 `True`)。同样 `num_stages=2`,关掉后:

```
26 × global_load_b128        (不再有 buffer_load)
 0 × 0x80000000              (哨兵消失)
61 × s_and_saveexec / s_cbranch_execz   (mask 改用 exec 收窄)
```

即无效 lane **根本不发射访存指令**,正确性不再依赖任何 SRD 配置。**这是区分两条路径的干净对照实验**:若 `AMDGCN_USE_BUFFER_OPS=0 ATOM_K3_ATTN_RES_NS=2` 端到端不再 fault,即可坐实 fault 出自 buffer ops 的 mask 实现。该验证需要一次全量长跑,**目前尚未完成**(机器已释放),是下一步最该做的事。

## 7. 想请 Triton / 编译器侧确认的问题

1. `BufferOpsEmitter` 在 gfx1250 上发出的 `num_records = 0xffffff`,与 `BufferOpsEmitter.h` 注释的 `max_int_32` 设计不一致 —— 是刻意的,还是 45-bit `num_records` 打包路径上的缺陷?
2. 为什么 `num_stages` 会影响 SRD 的构造方式(`ns1` 无 `0xffffff` 立即数,`ns2` 有)?这条耦合是否正是流水线与 fault 相关的真实原因?
3. GFX12+ 上 `OOB_select` 被 `lowerPointerAsRsrcIntrin()` 抹掉后退化为 `structured`,"stride=0 时两模式等价"这个假设在 gfx1250 硅上是否确实成立?若不成立,哨兵 `0x80000000` 会被真实发射,mask 即整体失效。
4. 用"发射越界访存 + 靠 SRD 丢弃"来实现 mask,相比 exec 收窄少了一层硬保障。gfx1250 上是否建议默认关闭 buffer ops,或对 `tl.arange` 造出的 2 的幂 padding 轴做特殊处理?
5. ROCm 侧推荐的访存越界检测手段?本次定位只能靠 launch 前后成对 `synchronize` 夹逼,成本很高。注意到源码树里有 `ConSanAMD.cpp` 与 `test_address_sanitizer.py`,这个 sanitizer 是否可用于此场景?

## 8. 复现材料(完整源码,无需查阅其他文件)

三份材料按"从最稳定到最不稳定"排列:**§8.2 的 IR dump 是稳定可复现的静态证据**(纯编译,不依赖运行时布局,几秒钟出结果,推荐从这里开始);§8.3 的 op 级 reproducer 目前**尚未**在干净进程里复现出 fault(原因见该节);端到端生产跑是目前唯一 100% 必崩的路径。

### 8.1 出问题的 kernel(生产源码,已含现有修复)

来自 `atom/models/kimi_k3_fused.py`。语义:每个 program 处理一行 `t`,对 `Bp = B+1` 个候选各做 rmsnorm,`score = <normed, score_weight>`,在 `Bp` 轴上 softmax,再加权求和。候选 `0..B-1` 来自 `block_residual`,候选 `B` 来自 `prefix_sum`。**崩溃命中的是第一个 H 循环**。

```python
@triton.jit
def _attn_res_fused_kernel(
    br_ptr,      # block_residual [T, B, H]  bf16
    ps_ptr,      # prefix_sum     [T, H]     bf16
    sw_ptr,      # score_weight   [H]        bf16
    y_ptr,       # out            [T, H]     bf16
    hs_ptr,      # hidden_states  [T, H]     bf16
    pref_ptr,    # prefix_out     [T, H]     bf16
    B, Bp, H, eps,
    stride_br_t, stride_br_b, stride_ps_t, stride_yt, stride_hs_t, stride_pref_t,
    BP: tl.constexpr,          # Bp padded to a power of 2 (vectorized candidate axis)
    BLOCK_H: tl.constexpr,
    NS: tl.constexpr,          # num_stages for the H-loop software pipeline
    DO_ADD: tl.constexpr,      # fold prefix += hidden_states on-load
    WRITE_PREF: tl.constexpr,  # write the (summed) prefix back to pref_ptr
):
    t = tl.program_id(0)
    b_idx = tl.arange(0, BP)
    b_mask = b_idx < Bp
    is_last = b_idx == B                      # prefix_sum candidate
    # BP rounds Bp=B+1 up to a power of 2, so lanes b_idx >= B name rows
    # block_residual does not have. Their values are never used (masked on load;
    # the b_idx==B lane takes ps via is_last), but the *address* must still be in
    # bounds. Clamping is addressing-only -- masked lanes keep resolving to
    # other=0.0.  <-- 见 §4;此 clamp 已上线,但 fault 依旧
    b_safe = tl.minimum(b_idx, tl.maximum(B - 1, 0))
    br_base = t * stride_br_t + b_safe * stride_br_b        # [BP]
    ps_base = t * stride_ps_t

    acc_sq = tl.zeros((BP,), dtype=tl.float32)
    acc_dot = tl.zeros((BP,), dtype=tl.float32)
    for h0 in tl.range(0, H, BLOCK_H, num_stages=NS):       # <-- 崩溃命中此循环
        cols = h0 + tl.arange(0, BLOCK_H)
        h_mask = cols < H
        br = tl.load(
            br_ptr + br_base[:, None] + cols[None, :],
            mask=(b_idx < B)[:, None] & h_mask[None, :],    # lane b_idx==3 恒为假
            other=0.0,
        ).to(tl.float32)
        ps = tl.load(ps_ptr + ps_base + cols, mask=h_mask, other=0.0).to(tl.float32)
        if DO_ADD:
            ps += tl.load(hs_ptr + t * stride_hs_t + cols, mask=h_mask, other=0.0).to(
                tl.float32
            )
        if WRITE_PREF:
            tl.store(                                       # 循环体内的 store
                pref_ptr + t * stride_pref_t + cols,
                ps.to(pref_ptr.dtype.element_ty),
                mask=h_mask,
            )
        v = tl.where(is_last[:, None], ps[None, :], br)     # [BP, BLOCK_H]
        sw = tl.load(sw_ptr + cols, mask=h_mask, other=0.0).to(tl.float32)
        acc_sq += tl.sum(v * v, axis=1)                     # [BP]
        acc_dot += tl.sum(v * sw[None, :], axis=1)          # [BP]

    rstd = 1.0 / tl.sqrt(acc_sq / H + eps)
    scores = tl.where(b_mask, rstd * acc_dot, float("-inf"))
    scores = scores - tl.max(scores, axis=0)
    probs = tl.exp(scores)
    probs = probs / tl.sum(probs, axis=0)                   # [BP], softmax over Bp

    for h0 in tl.range(0, H, BLOCK_H, num_stages=NS):       # 第二遍:加权求和
        cols = h0 + tl.arange(0, BLOCK_H)
        h_mask = cols < H
        br = tl.load(
            br_ptr + br_base[:, None] + cols[None, :],
            mask=(b_idx < B)[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        ps = tl.load(ps_ptr + ps_base + cols, mask=h_mask, other=0.0).to(tl.float32)
        if DO_ADD:
            ps += tl.load(hs_ptr + t * stride_hs_t + cols, mask=h_mask, other=0.0).to(
                tl.float32
            )
        v = tl.where(is_last[:, None], ps[None, :], br)
        out = tl.sum(probs[:, None] * v, axis=0)            # [BLOCK_H]
        tl.store(y_ptr + t * stride_yt + cols, out.to(y_ptr.dtype.element_ty), mask=h_mask)
```

launch 侧(同文件)按 token 数选配置,`T > 256` 落到 catch-all:

```python
_ATTN_RES_NS = int(os.getenv("ATOM_K3_ATTN_RES_NS", "1"))   # 出问题时是 2
_ATTN_RES_BLOCK_H = 1024
_ATTN_RES_CATCHALL = (False, 1, _ATTN_RES_NS, 4)            # (split, S, num_stages, num_warps)

_attn_res_fused_kernel[(T,)](
    block_residual, prefix_sum, score_weight, y, add_hidden, prefix_out,
    B, Bp, H, eps,
    block_residual.stride(0), block_residual.stride(1),
    prefix_sum.stride(0), y.stride(0), add_hidden.stride(0), prefix_out.stride(0),
    BP=triton.next_power_of_2(Bp), BLOCK_H=_ATTN_RES_BLOCK_H,
    NS=ns, DO_ADD=add_hidden is not None, WRITE_PREF=add_hidden is not None,
    num_warps=nw,
)
```

### 8.2 IR dump 脚本(稳定可复现,推荐从这里开始)

通过生产入口编译该 kernel,constexpr 与崩溃现场一致(`BP=4, BLOCK_H=1024, DO_ADD=True, WRITE_PREF=True`)。`H` 是运行时参数、不影响代码生成,故小 `T` 即可,只占十几 MB 显存、几秒钟跑完。完整源码:

```python
"""Dump the attn-res fused kernel's Triton IR + AMDGCN at num_stages=1 vs 2."""

import os
import sys

import torch

# T > 256 hits the catch-all config, i.e. the two-pass pipelined kernel that
# faults in production. Keep this above the largest bucket in _ATTN_RES_CONFIGS.
T = int(os.environ.get("DUMP_T", "300"))
B = int(os.environ.get("DUMP_B", "3"))
H = int(os.environ.get("DUMP_H", "7168"))


def main() -> int:
    from atom.models.kimi_k3_fused import _ATTN_RES_NS, _apply_attn_res_impl

    dev = "cuda"
    dt = torch.bfloat16
    prefix_sum = torch.randn(T, H, device=dev, dtype=dt)
    block_residual = torch.randn(T, B, H, device=dev, dtype=dt)
    score_weight = torch.randn(H, device=dev, dtype=dt)
    add_hidden = torch.randn(T, H, device=dev, dtype=dt)

    print(f"NS={_ATTN_RES_NS} T={T} B={B} H={H} dump={os.environ.get('TRITON_DUMP_DIR')}")
    y, pref = _apply_attn_res_impl(
        prefix_sum, block_residual, score_weight, 1e-6, add_hidden=add_hidden
    )
    torch.cuda.synchronize()
    print(f"ok y={tuple(y.shape)} finite={bool(torch.isfinite(y).all())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

跑法:

```bash
for NS in 1 2; do
  ATOM_K3_ATTN_RES_NS=$NS TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 \
    TRITON_DUMP_DIR=/tmp/ir_ns$NS python3 my_script/dump_attn_res_ir.py
done

# buffer ops 对照(§6.4)
ATOM_K3_ATTN_RES_NS=2 AMDGCN_USE_BUFFER_OPS=0 TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 \
  TRITON_DUMP_DIR=/tmp/ir_ns2_nobuf python3 my_script/dump_attn_res_ir.py
```

核对本文结论的三条命令:

```bash
# 1. 访存指令种类与数量(§6 表格)
grep -oE 'buffer_load_[a-z0-9]+|global_load_[a-z0-9]+' <dump>/_attn_res_fused_kernel.amdgcn | sort | uniq -c
# 2. 哨兵 voffset(§6.1)
grep -c '0x80000000' <dump>/_attn_res_fused_kernel.amdgcn
# 3. num_records 立即数,仅 ns=2 出现(§6.2)
grep -n 's_mov_b32 s[0-9]*, 0xffffff' <dump>/_attn_res_fused_kernel.amdgcn
```

### 8.3 op 级 reproducer(完整源码)

自包含,只依赖 torch + triton,**不 import 任何 ATOM 模块**。镜像生产 kernel 第一遍的形态:同样的多指针 load、同样的循环内 store、同样的 mask。`num_stages`、是否 clamp 寻址、循环内是否 store、是否把分配逼到显存尾部都是开关。

```python
"""Reproducer for the attn-residual aperture violation (gfx1250 / ROCm Triton)."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import triton
import triton.language as tl


@triton.jit
def attn_res_first_pass(
    br_ptr, ps_ptr, hs_ptr, pref_ptr, sw_ptr, acc_ptr,
    B, H, stride_br_t, stride_br_b, stride_t,
    BP: tl.constexpr,             # B+1 rounded up to a power of 2
    BLOCK_H: tl.constexpr,
    NS: tl.constexpr,             # num_stages of the H-loop software pipeline
    CLAMP: tl.constexpr,          # keep the candidate-axis addressing in bounds
    STORE_IN_LOOP: tl.constexpr,  # mirror the production kernel's WRITE_PREF
):
    t = tl.program_id(0)
    b_idx = tl.arange(0, BP)
    is_last = b_idx == B
    # Lanes b_idx >= B name rows br does not have. Masked off on load, so CLAMP
    # only affects whether the address stays in bounds.
    if CLAMP:
        b_addr = tl.minimum(b_idx, tl.maximum(B - 1, 0))
    else:
        b_addr = b_idx
    br_base = t * stride_br_t + b_addr * stride_br_b
    acc_sq = tl.zeros((BP,), dtype=tl.float32)
    acc_dot = tl.zeros((BP,), dtype=tl.float32)
    for h0 in tl.range(0, H, BLOCK_H, num_stages=NS):
        cols = h0 + tl.arange(0, BLOCK_H)
        h_mask = cols < H
        br = tl.load(
            br_ptr + br_base[:, None] + cols[None, :],
            mask=(b_idx < B)[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        ps = tl.load(ps_ptr + t * stride_t + cols, mask=h_mask, other=0.0).to(tl.float32)
        ps += tl.load(hs_ptr + t * stride_t + cols, mask=h_mask, other=0.0).to(tl.float32)
        if STORE_IN_LOOP:
            tl.store(
                pref_ptr + t * stride_t + cols,
                ps.to(pref_ptr.dtype.element_ty),
                mask=h_mask,
            )
        v = tl.where(is_last[:, None], ps[None, :], br)
        sw = tl.load(sw_ptr + cols, mask=h_mask, other=0.0).to(tl.float32)
        acc_sq += tl.sum(v * v, axis=1)
        acc_dot += tl.sum(v * sw[None, :], axis=1)
    tl.store(acc_ptr + t * BP + b_idx, acc_sq + acc_dot)


def _fill_vram(reserve_bytes: int = 512 * 1024 * 1024) -> list[torch.Tensor]:
    """Consume most of VRAM so the tensors under test sit near a segment tail.

    Frees `reserve_bytes` back afterwards; the tensors under test then land in
    that hole, close to the end of what is actually mapped.
    """
    ballast: list[torch.Tensor] = []
    chunk = 512 * 1024 * 1024
    while chunk >= 4 * 1024 * 1024:
        try:
            ballast.append(torch.empty(chunk, dtype=torch.uint8, device="cuda"))
        except torch.OutOfMemoryError:
            chunk //= 2
    freed = 0
    while ballast and freed < reserve_bytes:
        freed += ballast.pop().numel()
    return ballast


def run_once(T, B, H, block_h, ns, warps, clamp, store_in_loop, fill, verbose=False):
    BP = triton.next_power_of_2(B + 1)
    torch.cuda.empty_cache()
    ballast = _fill_vram() if fill else []

    br = torch.randn(T, B, H, device="cuda", dtype=torch.bfloat16)
    ps = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    hs = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    pref = torch.empty(T, H, device="cuda", dtype=torch.bfloat16)
    sw = torch.randn(H, device="cuda", dtype=torch.float32)
    acc = torch.zeros(T * BP, device="cuda", dtype=torch.float32)

    if verbose:
        nbytes = br.numel() * br.element_size()
        end = br.data_ptr() + nbytes
        print(
            f"T={T} B={B} H={H} BP={BP} BLOCK_H={block_h} num_stages={ns} "
            f"num_warps={warps} clamp={clamp} store_in_loop={store_in_loop} fill={fill}"
        )
        print(f"  br  : ptr=0x{br.data_ptr():x} bytes={nbytes} end=0x{end:x}")
        print(f"        tail page-aligned: {end % 4096 == 0}, whole pages: {nbytes / 4096}")

    attn_res_first_pass[(T,)](
        br, ps, hs, pref, sw, acc,
        B, H, br.stride(0), br.stride(1), ps.stride(0),
        BP=BP, BLOCK_H=block_h, NS=ns, CLAMP=clamp, STORE_IN_LOOP=store_in_loop,
        num_warps=warps,
    )
    torch.cuda.synchronize()
    del ballast


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1166, help="rows (grid size)")
    ap.add_argument("--B", type=int, default=3, help="real rows on the padded axis")
    ap.add_argument("--H", type=int, default=7168, help="reduction length")
    ap.add_argument("--block-h", type=int, default=1024)
    ap.add_argument("--ns", type=int, default=2, help="num_stages of the H loop")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--no-clamp", action="store_true", help="address without clamping")
    ap.add_argument("--no-store-in-loop", action="store_true")
    ap.add_argument("--fill-vram", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    if not a.sweep:
        run_once(a.T, a.B, a.H, a.block_h, a.ns, a.warps, not a.no_clamp,
                 not a.no_store_in_loop, a.fill_vram, verbose=True)
        print("  -> completed without fault")
        return

    print(f"{'B':>3} {'BP':>3} {'ns':>3}  result")
    for B in range(1, 9):
        for ns in (1, 2, 3):
            BP = triton.next_power_of_2(B + 1)
            try:
                run_once(a.T, B, a.H, a.block_h, ns, a.warps, not a.no_clamp,
                         not a.no_store_in_loop, a.fill_vram)
                res = "ok"
            except Exception as exc:  # noqa: BLE001 - report and stop
                res = f"FAULT: {type(exc).__name__}: {str(exc)[:60]}"
            print(f"{B:>3} {BP:>3} {ns:>3}  {res}")
            if res != "ok":
                return  # the context is unusable after an aperture violation


if __name__ == "__main__":
    main()
```

跑法(默认即崩溃现场参数,已 clamp 寻址、循环内带 store):

```bash
python repro_attn_res_oob.py
python repro_attn_res_oob.py --ns 1              # 关掉流水线
python repro_attn_res_oob.py --no-clamp          # 恢复 §4 的越界寻址
python repro_attn_res_oob.py --no-store-in-loop  # 去掉循环内 store
python repro_attn_res_oob.py --fill-vram         # 把分配逼到显存尾部
python repro_attn_res_oob.py --sweep             # 扫 B=1..8 x num_stages=1,2,3
```

**这个脚本目前的局限,请先看这段**:它在干净进程里**尚未复现出 fault**。原因是 allocator 布局 —— 越界量小(§4 那处是 2048 字节),常被 PyTorch caching allocator 的 block padding 吸收而静默通过;只有当分配尾部恰好是 segment 边界、越界打到未映射页时才会 fault。生产上 `block_residual` 恰好是 12243 个整 4KB 页(尾部页对齐),所以那里必崩。脚本默认设 `expandable_segments:True`、并提供 `--fill-vram` 来提高命中率,但仍不稳定。

**因此推荐的复现顺序是**:先用 §8.2 的 IR dump 看静态证据(稳定、几秒、不依赖运行时布局),需要端到端确认时再用生产跑(`ATOM_K3_ATTN_RES_NS=2` + 并发 64 跑 GSM8K,数十秒内必崩)。

## 9. 目前的处置

1. **寻址 clamp**(三个 kernel 各一处):消除 `b_idx >= B` 的越界地址。结合 §6.2,buffer ops 不提供真实边界保护,这类隐患必须在上层修掉,应当保留。
2. **`num_stages` 默认降为 1**,即 `_ATTN_RES_NS = int(os.getenv("ATOM_K3_ATTN_RES_NS", "1"))`。`T > 256` 的 catch-all 与 `T <= 256` 的桶都改用它,可用环境变量覆盖以做对照。按代码注释,pipelining 的收益只在小 T 出现(大 T 靠 occupancy 就能掩盖访存延迟),而小 T 走 split 路径,故性能代价预期很小。**已验证**:全量 GSM8K 1319 题跑通,47 分钟 0 fault,精度 0.9598(基线 0.9613,差 2 题,在 ±0.0054 标准误内)。
3. **保留一条 unfused 退路**:`ATOM_K3_ATTN_RES_FUSED=0` 切回 pre-fusion 路径(候选轴为 grid 维度、H 循环不带流水线),代价是多一次 `[T, Bp, H]` 物化。当前默认走融合路径,因为 `num_stages=1` 已足以规避。
4. **待做**:`AMDGCN_USE_BUFFER_OPS=0 ATOM_K3_ATTN_RES_NS=2` 的端到端验证(§6.4),用以坐实根因。

## 10. 附:上一版实现为何没有这个问题

同一功能的旧实现把候选轴放在 **grid 维度**,且 H 循环不带流水线:

```python
    t = tl.program_id(0)
    b = tl.program_id(1)                       # grid = (T, Bp)
    base = t * stride_t + b * stride_b
    for h0 in range(0, H, BLOCK_H):            # 无 num_stages
```

`program_id(1)` 精确落在 `[0, Bp)`,既没有 2 的幂 padding(不会为不存在的行算地址,也就不会有恒为假的哨兵 lane),也没有软件流水线。新实现把候选轴改为寄存器内的 `tl.arange(0, BP)` 并引入 `num_stages`,换来了性能(softmax 与加权和留在寄存器,省一次 HBM 往返),同时引入了这两个变量。
