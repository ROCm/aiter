# EP32 M=4096 优化实验（2026-11-12）

2026-11-13 当前状态：默认使用BM80 G1 + variable-width native G2 + 独立reduce。最后一次有效原bench默认入口为FC1 68.881 µs、FC2 41.081 µs、合计110.081 µs / 4.7830 TB/s，目标未达。配对曾达到5.1157T，仍不能替代默认验收。G1 B scale NT已撤回。当前实现及正确性见[g2var/README.md](g2var/README.md)，当前默认指标见[g2var/default_summary.json](g2var/default_summary.json)。以下章节保留当时的BM64/BNT atomic历史记录，不能据此判断当前默认源码或性能。

已保留 B-LDS 权重 DMA 的 NT 提示。当前 cx/aiter 相对 HEAD 2ee6af1df，仅 mxmoe_gemm2_ep32.py 有修改：B DMA 的 aux 改为 Python 整数 2，内核名增加 bnt；三槽 A/B LDS、提前两步的流水、MFMA 和 BF16 atomic 输出保持原算法。NT 只用于 B，A 保留 temporal 策略。

默认入口最后实测（HIP4/MI355X，原始 amd_moe_bench_share/bench_moe.py，EP32/M4096/rank0，3 次预热、30 个样本、512 MiB 只读 flush）：

| 指标 | 耗时 µs | BW TB/s |
|---|---:|---:|
| FC1 | 68.221 | 5.007 |
| FC2 | 40.740 | 4.539 |
| FC1+FC2 | 109.082 | 4.827 |

带宽使用脚本实际模型字节数：FC1 341592672、FC2 184921920、合计 526514592。合计耗时是逐样本相加后取中位数。FC2 的 5 TB/s 门槛为 36.984 µs，当前仍差 3.756 µs；合计门槛 105.303 µs，当前仍差 3.779 µs。目标未达成。

下表每一行是独立的同进程交替配对，各实现 3 次预热、30 个测量样本，保持原脚本的均衡路由和 512 MiB flush。只能在行内比较；不同捕获的绝对时间不同。

| 方案 | 同次原版 FC2 µs | 候选 FC2 µs | 候选 FC2 TB/s | 候选合计 TB/s |
|---|---:|---:|---:|---:|
| route 归并 / AcqRel | 45.181 | 324.263 | 0.570 | 1.341 |
| route 归并 / Monotonic | 45.020 | 85.681 | 2.158 | 3.425 |
| B 权重重排 | 44.221 | 44.301 | 4.174 | 4.646 |
| fx.Int32 aux / 无效 NT 提示 | 44.861 | 44.441 | 4.161 | 4.635 |
| B DMA NT / ISA 位补丁 | 42.460 | 40.660 | 4.548 | 4.863 |
| B DMA NT / Python 整数 aux | 42.141 | 40.761 | 4.537 | 4.828 |

源码 NT 配对 42.1405 → 40.7605 µs，FC2 快约 3.3%；另一个 ISA NT 配对快约 4.2%。最终默认入口使用实际生成 NT 的源码，未使用 ISA 缓存补丁。

route 归并使用真实本地 route 列表，单 route 直接 store，多 route 写 sorted-row BF16 暂存，并由最后到达者按 slot 顺序逐次 BF16 求和。AcqRel 版本生成 10 条 buffer_wbl2；其移除后从 324 µs 降至 85.7 µs，但仍显著慢于原版，因此未保留到默认路径。B 权重重排将每步四个 J16 子块的间隔由 24 KiB 改为相邻 1 KiB，正确但没有收益。这些具体实现已失败，未出现新的理由时不要重复。

NT 接口发现：FlyDSL 0.5.0 的 raw_ptr_buffer_load_lds 的 aux 是编译期属性，只接受 Python int 或 MLIR Attribute；fx.Int32(2) 被置为 None。该无效提示捕获的 2937 条指令与原版完全相同，0.9% 时间变化不能归因于 NT。真实 gfx950 语法为 nt lds；ISA 位补丁设置 B DMA 的 NT 位、保留 A DMA，随后共享 cache 按原始字节恢复并校验。最后用 Python int 修正源码，生成 96 条带 nt lds 的 B DMA。

正确性：原始 2ee6af1df 内核快照为 cx/g2route/baseline.py。当前均衡路由有 2046 条 route、1650 个活跃 token；本地 route 数直方图为 0:2446、1:1313、2:287、3:42、4:7、5:1。route 归并与原版逐专家贡献按 slot 顺序 BF16 求和的参考逐位一致，3 次重放和 arrival 计数均通过。B 重排及真实 NT 的全部 28 个专家隔离贡献与原版逐位一致，所有 <=2-route token 的完整输出也逐位一致；>=3-route token 保留 BF16 atomic 顺序差异，未声称完整输出逐位一致。

最终默认 trace 有 33 次 _blds2_bnt dispatch；metadata complete、output_finite=true。代码对象的 96 条 B DMA 有 NT，A DMA 没有增加 NT。最终默认版本与已完成上述正确性检查的 NT 源码配对版本，其规范化 2937 条 ISA 指令完全一致。资源保持原版：零 scratch、79872 B LDS；rocprof 的 VGPR_Count=128 是该工具版本减半的字段，不将其误报为实际仅 128 VGPR。

最终源码补丁：experiments_ep32_20261112/final_bnt.patch。完整实验源码及相对 2ee6af1df 的实验补丁存于 experiments_ep32_20261112/source 和 experiment.patch；失败实验没有留在默认 aiter 代码路径。g2route/check.py、paired_bench.py、run_pair.sh 是配套完整实验源码的工具，不能在当前精简 NT 工作树直接运行；回放旧实验需恢复对应实验 checkout 和辅助模块。当前默认复现只需：

```bash
cd /root/workspace/workflow_34/moonshot_ep32_moe_v2/cx
bash run_ep32_baseline.sh 4
```

脚本每次重新确认 GPU 空闲并生成独立结果目录。Python 使用 /tmp/aiter_ep32_branch_review_20261112/bin/python，FlyDSL 0.5.0，cache 使用 cc/wt_cache/flydsl050。

最终数据目录：/root/workspace/workflow_34/moonshot_ep32_moe_v2/cx/results/ep32_m4096_20261112_162404_hip4

配对捕获：

- route 归并 / AcqRel: /root/workspace/workflow_34/moonshot_ep32_moe_v2/cx/g2route/amd_pair_20261112_154448_hip4/bench/paired_summary.json
- route 归并 / Monotonic: /root/workspace/workflow_34/moonshot_ep32_moe_v2/cx/g2route/amd_pair_20261112_155042_hip4/bench/paired_summary.json
- B 权重重排: /root/workspace/workflow_34/moonshot_ep32_moe_v2/cx/g2bpack/amd_pair_20261112_155809_hip4/bench/paired_summary.json
- fx.Int32 aux / 无效 NT 提示: /root/workspace/workflow_34/moonshot_ep32_moe_v2/cx/g2bnt/amd_pair_20261112_160257_hip4/bench/paired_summary.json
- B DMA NT / ISA 位补丁: /root/workspace/workflow_34/moonshot_ep32_moe_v2/cx/g2bnt/amd_pair_20261112_161445_hip4/bench/paired_summary.json
- B DMA NT / Python 整数 aux: /root/workspace/workflow_34/moonshot_ep32_moe_v2/cx/g2bnt/amd_pair_20261112_162117_hip4/bench/paired_summary.json

仅测试指定 amd bench 的 EP32/M4096 点。未修改 cc/aiter 源码；未 commit、未 push。


## FC1 后续实验（2026-11-12）

本轮FC1候选未接入默认入口。当前aiter工作树仍只有mxmoe_gemm2_ep32.py的既有NT改动；FC1与原版快照逐字节相同。正式bench和其他fmoe kernel未修改。

同进程正式对照：完整K滚动控制69.161→69.261 µs；K9+5/128-bit partial 69.161→73.2205 µs；resident辅助组69.441→75.241 µs；A-tail probe+B scale提前预取68.8405→69.5605 µs。分摊两版均较慢，scale提前预取也未获益。所有有效候选的实际路由FP8 payload（6,285,312元素）与scale均三次重放逐位一致。未接入慢候选，目标5 TB/s未达。

诊断发现：新滚动K循环的早期无等待版本存在A DMA同步错误。workgroup barrier不等于DMA完成；阻止常量传播不能单独解决问题。明确vmcnt(0)或last-A DMA的volatile LDS probe修复结果。无等待探针废弃，不作为可信优化基线。

旧默认入口本轮复测HIP5：68.9805/44.2205/112.9615 µs，合计4.6610 TB/s；此前HIP4为4.8268 TB/s。源码、配置、正式bench哈希不变，跨卡/跨捕获差异不归因于显存。用户取消了显存因果对照，没有执行。原脚本prepare/reduce时间不进入FC1+FC2 BW分母，按用户最新指示没有修改排序/copy/初始化等部分。

全部实现、原始日志、正确性和ISA证据见g1stream/README.md；源码归档g1stream/source/，默认kernel目录已清理。没有commit、没有push。
