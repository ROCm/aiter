# March TopK 代码说明

这个目录主要整理了我在 `60K FP32 -> top2048` 场景下做 TopK 优化时保留下来的三组代码：

- `baseline-carlus` 72us~73us
- `csdn`  64us
- `final-version` 42us~47us

当前默认的测试口径主要是：

- 输入规模：`N = 60000`
- 数据类型：`FP32`
- TopK：`K = 2048`
- 主要设备：`gfx942 / MI300X`

## 1. `baseline-carlus`

这一组是基准代码，来源是：

- `baseline-carlus/topk_per_row_kernels.cu`

它是原始的 AIR / Carlus 风格 `topk_per_row_kernels.cu` 实现，包含完整的 per-row TopK 逻辑，同时支持：

- `Prefill`
- `Decode`
- 单 block 路径
- 多 block 路径

为了更方便单独 benchmark 和分析，我又从这个文件里拆出了两个 standalone 版本：

- `baseline-carlus/topk_baseline_oneblock.cu`
  - 单线程块版本
  - 适合看 one-block 路径的性能和汇编

- `baseline-carlus/topk_baseline_multiblocks.cu`
  - 多线程块版本
  - 适合看 multi-block 路径、global histogram、software sync 等行为

因此，这一组可以理解为“基准实现 + 从基准实现拆出来的 oneblock/multiblocks benchmark 版本”。

## 2. `csdn`

- `csdn/topk_sort.cu`

这是一个尝试版本，不是最终采用的主线。

它的主要思路是：

- 使用较小位宽的 radix pass（4-bit）
- 在 pass 0 之后做一次 compact
- 后续 pass 只处理 compact 后的候选集，而不是每轮都扫描全量输入

这版的重点是尝试“通过一次 compact 来减少后续总访存量”，属于一个偏工程化的探索版本，和 baseline 的实现路线不完全相同。

## 3. `final-version`

这一组是最终保留下来的 one-block 优化版本，主要文件有两个：

- `final-version/topk_opt_prefill.cu`
- `final-version/topk_opt_prefill_decode.cu`

可以大致理解为：

- `topk_opt_prefill.cu`
  - 只保留 `Prefill` 相关逻辑/实例化
  - 这是最终的 prefill 优化版本
  - 当前测到的 latency 大约在 `47us`

  相对 baseline (`topk_baseline_oneblock.cu`) 的具体优化点：

  1. **[OPT3] `calc_bucket`：单指令位域提取**
     - baseline：`(twiddle >> start_bit) & mask`，编译器生成 shift + and 两条指令
     - 优化后：使用 `__builtin_amdgcn_ubfe` 内建函数，直接映射到 `v_bfe_u32` 单条 GCN 指令

  2. **[OPT3] `vectorized_process`：移除 batching 聚合，直接 atomicAdd**
     - baseline：使用 `acc` / `prev_bin_idx` 做 warp 内合并（如果连续元素落到同一个桶，就在本地累加后再一次性写入），逻辑复杂且引入额外分支
     - 优化后：移除 batching 逻辑，每个元素直接 `atomicAdd` 到 LDS histogram。在 11-bit（2048 桶）的场景下，碰撞率极低（约 1/2048），batching 几乎无收益，反而增加了指令数

  3. **[OPT3/4] `vectorized_process`：4x 展开宽加载 + load-compute 交织**
     - baseline：1x 宽加载（每次读一个 `fp32x4`），计算完再读下一个
     - 优化后：4x 展开，一次发射 4 个 `global_load_dwordx4`（共 512 字节 / 线程），在等待第一批数据返回的延迟窗口内发射后续加载请求，实现 load-compute 流水线交织，更好地隐藏 VMEM 延迟

  4. **[OPT5-A] `filter_and_histogram_for_one_block`：pass > 0 的 `!out_buf` 路径向量化**
     - baseline：pass > 0 使用 `for(i += blockDim.x)` 标量循环逐元素读取，编译器生成 `global_load_dword`（32 位标量加载）
     - 优化后：改用 `vectorized_process` 走 `global_load_dwordx4`（128 位宽加载），吞吐量提升 4x
     - 同时优化了 twiddle 计算：只调用一次 `twiddle_in` 得到完整 bits，再分别提取 prefix 和 bucket，避免重复 twiddle 开销

  5. **[OPT5-B] `last_filter`：`!in_idx_buf` 路径向量化**
     - baseline：使用 `for(i += blockDim.x)` 标量循环，生成 `global_load_dword`
     - 优化后：改用 `vectorized_process`，走 `global_load_dwordx4` 宽加载

  6. **[OPT5-C] `radix_topk_one_block_kernel`：禁用 compact buffer 乒乓**
     - baseline：使用 `set_buf_pointers` 在 pass 之间做 compact，将匹配 prefix 的元素拷贝到中间缓冲区，后续 pass 只扫描缓冲区中的候选集
     - 优化后：所有 pass 都直接读原始输入（`out_buf = nullptr`），走 `vectorized_process` 路径
     - 虽然每个 pass 都要扫描全量数据（60K），但宽加载 + 4x 展开的向量化路径比标量 compact 路径更快，且省去了 compact 本身的写入开销

- `topk_opt_prefill_decode.cu`
  - 在前一个版本基础上，增加了 `Decode` 相关逻辑/实例化与测试入口
  - 令人意外的是，这个版本里 `Prefill` 的 latency 反而下降到了大约 `42us`

## 4. 关于 47us 和 42us 的现象

目前这个现象是我觉得比较奇怪、也还在继续看的一个点。

表面上看：

- `topk_opt_prefill_decode.cu` 只是比 `topk_opt_prefill.cu` 多了 `Decode` 逻辑
- 按直觉，`Prefill` kernel 本身不应该因此直接快这么多

目前更合理的猜测是：

- 并不是显式算法逻辑又多了一条新的大优化
- 而是因为编译单元变了（例如多了 `Decode` 的模板实例化、调用路径、host 端入口）
- 最终导致 HIP / clang 对 `Prefill` 路径生成了不同的 device code
- 也就是说，`Prefill` latency 从 `~47us` 降到 `~42us`，更像是 **codegen 变化**，而不一定是肉眼可见的源码逻辑变化

这个问题目前还没有完全定论，后续如果继续深入，需要进一步对比：

- `Prefill` kernel 的 ISA / 汇编
- VGPR / SGPR 占用
- 调度与 waitcnt 排布

## 5. 总结

如果只想快速理解这个目录，可以这样看：

- `baseline-carlus`：基准代码，以及从基准里拆出来的单/多 block benchmark
- `csdn`：一个做过 compact 尝试的探索版本
- `final-version`：最终保留下来的 one-block 优化版本；其中 `prefill+decode` 版出现了“仅增加 decode 逻辑，但 prefill latency 反而更低”的现象，目前怀疑是编译生成代码变化导致
