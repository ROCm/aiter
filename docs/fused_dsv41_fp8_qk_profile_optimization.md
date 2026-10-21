# FP8 QK：ATT / PMC 优化与最终验证（2026-10-21）

最终候选：`dequant16_qkds_storeflip_h64late`。源码基线 `06ca613d93b1d19d7edb411f5dd79c26b6239c23`，
分支 `perf/dsv41-fp8-qk-profile-opt-20261021`，本轮提交未推送。

六组完整路径中，row fused 相对旧版降低 **20.34%–28.10%** 延迟。
H64 四组快于 PR 分开的完整路径；H128 仍慢 **5.85% / 8.10%**。
因此“全部形状 fused 快于 PR”的目标尚未达成。

## 同轮性能

单位 μs。负的相对 PR 延迟表示 fused 更快。

| T / H / S / K | 旧 fused row | 最终 row | 最终 paged | PR 全路径 | row 相对 PR 延迟 |
|---|---:|---:|---:|---:|---:|
| 560 / 64 / 8192 / 640 | 133.04 | 102.92 | 102.22 | 112.79 | -8.75% |
| 560 / 64 / 32768 / 640 | 136.31 | 103.69 | 103.32 | 115.38 | -10.13% |
| 960 / 64 / 8192 / 640 | 164.57 | 130.99 | 130.64 | 176.34 | -25.71% |
| 960 / 64 / 32768 / 640 | 166.59 | 132.71 | 131.28 | 178.52 | -25.66% |
| 320 / 128 / 8192 / 1152 | 216.66 | 162.73 | 163.11 | 153.74 | +5.85% |
| 320 / 128 / 32768 / 1152 | 235.33 | 169.21 | 169.47 | 156.54 | +8.10% |

MI355X / gfx950，`mi355-gpu-47`，`yzhou_atom_latest2`，HIP6；时钟未锁定。
PR #6675 固定为 `b92f3cc9eeff2b5a5c56741a15da003dc5640d2e`。
比较包含 Q RoPE、group32 Q quant、attention、inverse RoPE 和输出 quant，同一 FP8 KV 与 pinned fixture。
PR 为 Q prepare + attention + epilogue 三个 kernel，采用 PR row 与 fused row 对比。
fused 还写出 max/LSE；PR 未请求可选 LSE。排除 GEMM、cache writer、布局转换、metadata/分配和模型 forward。
原始 samples、平衡 graph 执行次序与 replay poisoning 检查保存在 benchmark JSON。
正式计时未开启 profiler，ATT/PMC 插桩时间不用于速度比较。

## 保留的修改及其依据

1. 提前预取下一轮 sparse index；paged index 保持 int32 到消费点。原始 scale words 跨过 QK，再在 raw commit 时处理。
   这解除旧 ATT 中 QK 前 PC 338796 `vmcnt(0)`、339108 `vmcnt(8)` 的早等待。
2. raw KV 使用每线程 4×16B 读取和 LDS 写入；`Words4` 以 scalar uint32 字段保持公共接口的 4B row 对齐约定。
   invalid/tail 从现有 zero row 读取，commit 写入 scale 127，score 有效性 mask 保留。
3. V 每次反量化 16 个元素，保留原 BF16 V 布局；PV 使用三级 V 预取。
4. QK 使用显式 `ds_read_b128` / `ds_read_u8` 与 immediate offset，保留手工 partial wait 和 scheduling fence。
   phase 1 中编译器每隔一组读取插入全等待；最终循环内完整等待由 5 次减到 1 次（最后 drain），partial wait 为 7 次。
   单独改等待能减少 sampled LDS wait，但总耗时收益不稳定；没有把等待降幅当作 kernel 加速比例。
5. V 的两次 BF16×8 store 对 `(lane & 4)` 两组使用相反半区顺序，最终字节布局不变，降低同一时刻的 bank 重叠。
   单独 storeflip 对照中 H128 约 165.05/171.33 → 162.81/169.18 μs；PMC LDSBankConflict 14.70 → 9.29。
6. 仅 H64 把 next raw commit 延后到 V dequant/softmax/O rescale 后，仍在 PV barrier 前。
   H128 保留 QK 后立即 commit。同轮候选对照中 H64 T560 105.35 → 102.83 μs、T960 136.08 → 131.24 μs；
   H128 162.32/169.32 与组合版 162.59/169.44 基本持平。选择基于 head 数，不按 benchmark 的 S/T 表派发。

QK 保持 scaled FP8 MFMA，PV 保持 BF16 MFMA，softmax/累加为 FP32，已有显式 FMA 保留。
数学顺序与 N32 分组保持不变；所有主要最终输出、scale、max/LSE 与旧版逐位一致。`fp8_qk=True` 继续为 opt-in。
生产修改仅在 `decode_fp8_qk.hpp` 与 `fused_dsv41_mxfp8.cu`，后者增加 commit schedule 模板参数与 head 派发。

## ATT / PMC 与仍未解决的部分

| H128 PMC | 旧 fused | phase 1 | 最终 fused | PR attention |
|---|---:|---:|---:|---:|
| SQ_INSTS_LDS | 12,154,880.000 | 10,772,480.000 | 10,772,480.000 | 5,186,560.000 |
| SQ_INSTS_VALU | 31,208,960.000 | 23,861,760.000 | 26,823,680.000 | 13,012,480.000 |
| SQ_INSTS_MFMA | 3,686,400.000 | 3,686,400.000 | 3,686,400.000 | 3,778,560.000 |
| SQ_INSTS_VMEM_RD | 1,006,080.000 | 650,240.000 | 650,240.000 | 645,120.000 |
| SQ_WAVES | 2,560.000 | 2,560.000 | 2,560.000 | 2,560.000 |
| MeanOccupancyPerCU | 4.878 | 5.162 | 5.091 | 3.007 |
| LDSBankConflict | 7.781 | 14.705 | 9.255 | 0.590 |
| lds_bytes | 68,096.000 | 68,096.000 | 68,096.000 | 147,968.000 |

PR attention 列只包含 attention，fused 列包含完整融合路径。比较 VALU 总量时需加上 PR 的 Q prepare 和 epilogue，
不能把 attention 单 kernel 的差异当作整条路径的额外开销。MFMA 动态工作量没有因本轮调度优化减少。
最终 VALU 为 26,823,680，高于 phase 1 的 23,861,760（约 +12.4%）；bank 调整的选择/地址操作有额外成本，
这是降低 bank 冲突的代价。最终 fused VALU 为 PR 三个 kernel 合计 25,116,160 的约 1.07 倍；
LDS 指令为 PR attention 的约 2.08 倍。不能仅按 bank 指标判断方案收益。

| H128 ATT：每条 sampled wave 的累计 cycles | 旧 fused | phase 1 | 最终 fused |
|---|---:|---:|---:|
| vmem_wait | 51,690.0 | 12,202.7 | 12,750.3 |
| lds_wait | 26,088.5 | 24,147.7 | 18,507.0 |
| lds_issue | 65,969.0 | 72,055.7 | 69,116.3 |
| valu_issue | 63,125.0 | 49,102.0 | 51,686.0 |
| mfma_issue | 16,312.5 | 17,958.3 | 16,492.0 |
| s_barrier | 11,115.0 | 6,768.3 | 5,869.7 |

ATT 按 sampled wave 数归一化；instruction latency 包含 issue/stall，不等于 kernel 墙钟占比或 GPU 利用率。
`LDSBankConflict` 是派生指标，不是冲突指令百分比。TCC HIT/MISS 无法单独证明 HBM 吞吐饱和，EA RDREQ 本机未返回。

剩余重点是 V 物化、PV 的 LDS 读与跨 wave 同步。PR 的 PV 按输出 D 分工并共享 P，使一份 V operand 供四个 head group 复用；
当前 fused 的 head 分工重复读取 V。这解释了相当一部分 LDS 指令差距，但改变分工还有 P/factor 交换及 epilogue 成本。
最终热点的 PC、指令和相邻 ISA 见 `final_stall_contexts.json`；QK 两种 commit schedule 的实际等待序列见 `qk_schedule.json`。
最终 H128：PV 前 barrier 为 PC 407448；QK 后 raw commit 的 VMEM wait 为 PC 405664；
QK 最后 drain 为 PC 405544，部分等待如 PC 405524 / 405436 仍可见。
相对 phase 1，sampled LDS wait 24,147.7 → 18,507 cycles/wave（约 -23.4%），
barrier 6,768.3 → 5,869.7，LDS issue 72,055.7 → 69,116.3；这些不是等比例墙钟收益。
消除源代码看起来多余的等待并不能单独消除 H128 的完整路径差距。

## 未采用的实验

- 显式 QK DS 无 immediate offset：额外地址 VALU 增多；immediate 版本恢复原地址计算量。两者 34 项测试通过。
- raw LDS XOR：34 项通过，无明显稳定收益；未与 storeflip 组合运行，不声称组合结果。
- 全 H128 共享 KV 的 8-wave CTA：34 项通过，但 H128 193/200 μs，慢于 163/169 μs，弃用。
- 跨 head PV 复用新版：P 只加载一次、每次仅保留一份 V，34 项通过且输出逐位一致。
  row 252 VGPR、0 AGPR、0 scratch、72192 B static LDS；H128 174.90/180.51 μs，慢于对照 162.44/169.21 μs，弃用。
  这次回退不能归因于 scratch spill；尚需分解 P/factor 交换、V load 串行等待和不均衡 epilogue。
- 所有 head 都 late commit：H64 改善，但 H128/S8192 163.03 → 165.85 μs 回退；因此只对 H64 使用。
- PV 四级预取、raw blocked layout：没有稳定收益。
- 每次 32 个 V 反量化：34 项通过、无 scratch，但比 16 个元素慢约 10–11%。
- N64：修复 byte-addressed LDS offset 后仍曾出现 NaN；PV inline MFMA 前加 `s_nop 3` 后通过 34 项与六组独立 oracle 检查。
  277 VGPR/21 AGPR、135680 B static LDS，H128 244/254 μs，弃用。N64 改变归一化分组，输出相对 N32 L2 约 0.0069；
  原 N32 等价检查失败记录保留，未把该数值变化合入最终候选。N64 native-index 衍生版仅写入生成器，未运行。

## 验证与复现

格式化后的源码用新 JIT 目录重编译：**253 项通过，0 skipped**。
六组主 benchmark 验证：输出/scale/max/LSE 与旧版逐位一致、row/paged 一致、独立 quantized-Q dense oracle、PR 误差和 graph replay。
另有 T1/T64/T1024 × H64/H128、S32768/K640 的六组 shared-cache 检查，通过且与旧版逐位一致。
24 个 FP8 specializations 均审计 scaled FP8 QK、BF16 PV、无 scratch；覆盖两种 commit schedule、Q 模式与 page 模式。
源码和最终 `.so` SHA256 在 `provenance.json`，归档 manifest 给出提交及逐文件 SHA256。

```text
repo: /home/jun_chen2_qle/yzhou_aiter/aiter-dsv41-fp8-qk-profile-opt-20261021
results: /home/jun_chen2_qle/yzhou_aiter/fp8-qk-profile-opt-20261021/release
binary: results/jit/final/module_fused_dsv41_mxfp8.so
```

环境：`HIP_VISIBLE_DEVICES=6 GPU_ARCHS=gfx950 MAX_JOBS=8 OMP_NUM_THREADS=4`，
`AITER_CK_DIR=/home/jun_chen2_qle/yzhou_aiter/aiter/3rdparty/composable_kernel`，PYTHONPATH 包含 repo 与 repo/op_tests。
GPU 工作串行。`full_benchmark.py final`、`shared.py`、`final_profiles.py` 为正式复现入口；需要新输出目录，避免覆盖既有证据。
主要 pinned fixture/adapter 依赖在 archive `evidence/dependencies` 中保留。
ATT 采集使用 rocprofv3、第三次 invocation、CU1、SE mask1、128 MiB、activity8。
本地旧 `att_review.zip` 是 phase 1（`wide_native_pv3_dequant16`）；新 `profiles.zip` 的 `release/final_*` 才对应本报告。
旧检查点、失败实验与阶段性 profile 均保留，便于独立复核。
