# FP8 QK：继续扩大 H64 优势并优化 H128（2026-10-21）

本轮保留 `h64_flags_prefetch`。基线是上一轮 `950eaa3f23411469db0c426f5876039f63594715`，
分支 `perf/dsv41-fp8-qk-pv-opt-20261021`，未推送。源代码、测试和 profiling 绑定最终二进制 SHA256。

H64 四组 row 相对上一版降低 3.32%–7.25% 延迟；
H128 两组降低 1.03%–1.40%。
H128 仍比 PR 完整路径慢 4.37% / 5.90%，尚未达到全部形状 fused 更快的目标。

## 同轮完整路径计时

单位 μs；负的相对 PR 延迟表示 fused 更快。

| T / H / S / K | 上一版 row | 本轮 row | 本轮 paged | PR 全路径 | row 相对 PR 延迟 |
|---|---:|---:|---:|---:|---:|
| 560 / 64 / 8192 / 640 | 102.79 | 95.34 | 95.10 | 113.10 | -15.70% |
| 560 / 64 / 32768 / 640 | 103.48 | 95.99 | 95.57 | 116.10 | -17.32% |
| 960 / 64 / 8192 / 640 | 130.76 | 125.49 | 124.10 | 178.19 | -29.58% |
| 960 / 64 / 32768 / 640 | 131.86 | 127.48 | 126.87 | 179.18 | -28.85% |
| 320 / 128 / 8192 / 1152 | 162.35 | 160.68 | 160.82 | 153.95 | +4.37% |
| 320 / 128 / 32768 / 1152 | 169.12 | 166.75 | 167.27 | 157.46 | +5.90% |

MI355X/gfx950，mi355-gpu-47，yzhou_atom_latest2，HIP6；时钟未锁定。
PR #6675 固定为 b92f3cc9eeff2b5a5c56741a15da003dc5640d2e。
同一份 FP8 KV 与 pinned fixture，包含 Q RoPE、group32 Q quant、attention、inverse RoPE 与输出 quant。
PR 三个 kernel 的完整路径与 fused row 对比；fused 另输出 max/LSE，PR 未请求可选 LSE。
排除 GEMM、cache writer、布局转换、metadata/分配和模型 forward。计时不启用 profiler。
benchmark JSON 保留平衡 graph 次序、原始 samples 和 replay poisoning 检查。

## 保留的三项修改

1. Q 直接按 scaled MFMA 的 lane 顺序准备。
   每个 lane 读取两个 16 元素片段；lane group 0/1、2/3 分别交换半组 amax，恢复同一个 group32 最大值。
   再置换一个 scale code 到原有的 scale lane 顺序。每个 K step 只交换两个 amax 和一个 code。
   raw Q 准备中的 ds_bpermute_b32 从 64 条减少到 12 条；prepared Q 直接读取最终片段，省掉数据重排。
   RoPE 后先舍入到 BF16、group32 exponent 计算和 FP8 转换保持一致。
2. 全有效 raw tile 的 score mask 快速路径。
   每个 wave 用 ballot 检查其 8 行是否全有效，把结果写在现有行 flag 的 bit 1，bit 0 保留行有效性。
   QK 读取前四行的 flags（各来自一个 wave），仅在四个 bit 1 全为 1 时跳过逐 score mask 读取。
   含无效索引或尾部不完整的 tile 继续执行原 mask；无需增加 LDS 空间。
   H64 把这次 flag 读取提前到 QK 之前，由第一组 K 操作数的 lgkmcnt(3) 一并等待其完成，避免 QK 后单独等待。
   H128 的同轮对照略有回退，保留 QK 之后读取 flag。
3. 仅 H64 缓存当前 V 的 scale words。
   从现有 raw prefetch 结果保留四个 words 到下一轮，在 V 反量化时取对应字节。
   不在 QK 前提前打包 next scale，避免重新引入 VMEM 早等待。
   H128 的单独对照没有稳定收益，最终仍使用原 LDS scale 读取。私有模板参数改名 H64Schedule，统一选择 H64 的 commit、scale 和 flag 调度。

QK 仍为 scaled FP8 MFMA，PV 为 BF16 MFMA，softmax/累加为 FP32。
原来的 QK partial wait、V storeflip、三级 PV 预取和 H64 late commit 保留，默认仍为 opt-in `fp8_qk=True`。
本轮没有改变归一化分组、计算精度或输出契约。

## 正确性与资源

最终格式化源码重新构建后，253 项回归全部通过，0 skipped。
覆盖 H64/H128、row/paged（page=0/7/32/128）、raw/prepared Q、跨 32 的尾块、负数/越界索引、零行及 graph replay。
原 fixture 含每 37 个位置的无效索引和整块无效索引，覆盖非首行无效及有效/无效 tile 交替，防止快速 mask 误判。
六组主 benchmark 与六组 shared-cache（T1/T64/T1024 × H64/H128）检查通过，
最终输出、scale、max/LSE 与上一版逐位一致；独立 quantized-Q dense oracle 和 PR 误差标准保持不变。
24 个 specialization 全部包含 scaled FP8 QK 与 BF16 PV，均无 scratch。
最终 raw-Q row 的 H64/H128 分别使用 250/245 VGPR，paged128 为 249/244 VGPR；均为 0 AGPR。
static LDS 为 67,840 B，PMC runtime LDS 为 68,096 B，保持原分配规模。

## PMC / ATT

| Counter | H64 上一版 | H64 本轮 | H128 上一版 | H128 本轮 |
|---|---:|---:|---:|---:|
| SQ_INSTS_LDS | 5,340,160.000 | 4,910,080.000 | 10,772,480.000 | 10,362,880.000 |
| SQ_INSTS_VALU | 14,940,800.000 | 14,367,360.000 | 26,823,680.000 | 25,228,800.000 |
| SQ_INSTS_MFMA | 1,792,000.000 | 1,792,000.000 | 3,686,400.000 | 3,686,400.000 |
| SQ_INSTS_VMEM_RD | 353,920.000 | 340,480.000 | 650,240.000 | 634,880.000 |
| MeanOccupancyPerCU | 4.356 | 4.455 | 5.091 | 5.149 |
| LDSBankConflict | 6.998 | 7.094 | 9.255 | 9.464 |

| H128 ATT cycles / sampled wave | 上一版 | 本轮 |
|---|---:|---:|
| vmem_wait | 12,750.3 | 13,389.0 |
| lds_wait | 18,507.0 | 24,590.5 |
| lds_issue | 69,116.3 | 73,043.5 |
| valu_issue | 51,686.0 | 50,026.5 |
| s_barrier | 5,869.7 | 6,263.0 |

H64 的 baseline PMC 为本轮补采；H128 的 baseline 来自上一轮最终二进制的已存档 trace。
ATT 按 sampled wave 归一化，其累计 instruction latency 不是 kernel 墙钟占比。
本次 final ATT 采到 8 个 wave，baseline 采到 12 个；两次 wave 的运行环境不同，归一化后也不能直接视为稳定的周期节省。
同一条 wait 同时包含 vmcnt 和 lgkmcnt 时，汇总脚本归入 VMEM，不能用该汇总进一步分离两类等待。
LDSBankConflict 是派生指标，不是冲突指令百分比；TCC hit/miss 也不能单独证明 HBM 饱和。
热点的 PC 与相邻指令保存在 final_stall_contexts.json；原始 ATT、decoder JSON 和实际引用 code objects 单独归档。

H64 的独立 LDS wait 汇总从 baseline 15,836.3，经第一版组合 13,358.7，降至最终 12,239.0 cycles/wave。
最终 trace 中 PC 346280 是 flag ds_read_b128，随后发出两组 K 操作数读取，PC 346348 为 lgkmcnt(3)，再执行第一条 scaled MFMA。
这证实预期的发指令次序；总体收益以未开启 profiler 的完整路径计时为准。

H128 最终热点的十进制 PC：404964 为 PV 前 barrier，403260 为下一 tile raw commit 的 vmcnt(0)，
403008 为 QK 后 flag 读取的 lgkmcnt(0)，405812 为 PV 完成后的 barrier。
H128 的 LDS 指令由 10,772,480 降至 10,362,880，但仍约为 PR attention 本体 5,186,560 的两倍；
这不是严格相同的 kernel 工作范围，fused 还包含 Q 准备与输出量化，完整路径计时见前表。
LDSBankConflict 从 9.255 变为 9.464，没有改善。

剩余 H128 的优化重点仍是 PV 的 V 物化、重复 LDS 读取和同步。PR 按输出 D 分工，使 V 在四个 head group 之间复用；
fused 当前的 head 分工重复读取 V，但换分工需要 P/factor 交换及不同的 epilogue，减少 LDS 指令不等于总耗时必然降低。
下一步需要针对 P/factor 交换成本及 V 的 LDS 布局重新设计，当前数据不支持仅靠继续调整 wait 或移动 barrier 就能补齐差距。

## 实验筛选记录

- readflip8：把半块翻转提前到 raw FP8 LDS 读取，避免 BF16 向量选择；34 项通过，H128 约 164.86/171.36 μs，回退。
- predequant_barrier：让下一轮 QK 先执行，再同步后复用 V；34 项通过，H128 约 164.03/170.84 μs，无稳定收益。
- storebranch：分支控制两次 V store 的顺序；34 项通过，但 H128 约 180.18/187.28 μs，H64 也回退。
- crosspipe2：P 保持驻留、双 V 预取并在四个 head group 复用。34 项通过，252 VGPR/0 AGPR/0 scratch（row），72192 B static LDS。
  H128 约 170.67/176.93 μs，比上一轮串行 V 读取版本改善，但仍慢于当前 head 分工；未保留。
- cached_scales：34 项通过；H64 T560 103.18→99.86 μs、T960 132.89→130.26 μs，H128 约持平，最终只对 H64 使用。
- qdirect：34 项通过，输出逐位一致；H128 162.48/169.38→160.88/167.53 μs，H64 103.00/132.65→100.84/128.64 μs。
- valid_fast：34 项通过；H64 102.70/132.11→100.60/129.53 μs，H128 162.38/169.29→161.67/168.53 μs。
- flags_prefetch：第一版组合的 H128 ATT 显示 flag 判断新增 LDS 等待（采样 LDS wait 从 18,507 增至 25,192.7 cycles/wave）。
  将 flag 提前到 QK 前，34 项通过；与组合版同轮对照，H64 97.62/126.51→94.95/125.05 μs，
  H128 160.35/166.82→160.83/167.45 μs 略回退。最终仅 H64 提前读取。

cached_scales 第一次计时因实验驱动脚本命名为 queue.py、遮蔽 Python 标准库而在导入 torch 时失败；
已改名为 pipeline_jobs.py，后续计时重跑成功。该失败未启动 GPU 测量，不是 kernel 正确性失败。
各候选的二进制哈希、输出等价检查、原始计时与状态保留在 trial_summary.json 及 evidence 归档。

## 复现

```text
repo: /home/jun_chen2_qle/yzhou_aiter/aiter-dsv41-fp8-qk-pv-opt-20261021
results: /home/jun_chen2_qle/yzhou_aiter/fp8-qk-pv-opt-20261021/release
binary: results/jit/final/module_fused_dsv41_mxfp8.so
```

环境为 HIP_VISIBLE_DEVICES=6、GPU_ARCHS=gfx950、MAX_JOBS=8、OMP_NUM_THREADS=4，
AITER_CK_DIR 指向 /home/jun_chen2_qle/yzhou_aiter/aiter/3rdparty/composable_kernel，PYTHONPATH 包含 repo 和 repo/op_tests。
GPU 工作串行。正式入口是 full_benchmark.py final、shared.py、final_profiles.py；使用新输出目录避免覆盖旧证据。
ATT 使用 rocprofv3、第三次 invocation、CU1、SE mask1、128 MiB buffer、activity8。
baseline 的含义是 950eaa3f2，对应上一份 fp8_qk_profile_opt_20261021 的 release/final。
manifest 记录最终提交与逐文件 SHA256；provenance.json 记录实际编译源码与二进制哈希。
