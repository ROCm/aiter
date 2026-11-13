# EP32 M4096 BM80 native GEMM2：v2 工作树记录

本分支保存 Kimi-K3 A8W4、H3584/I3072、EP32/local28、M4096 的当前实现和待验证优化资料。原 `amd_moe_bench_share/bench_moe.py` 及 parser 未修改。**本工作树的默认入口尚未达到 FC1+FC2 5 TB/s。**

## 默认实现

现有 CSV 只更新 M4096/topk16、17 两行的 block_m 和 kernel 名称。GEMM1 支持 BM80 排序，并处理 80 行块跨 32 行 scale 块的对齐，保持原 SiTUv2、BF16 materialization 和 MXFP8 量化。GEMM1 B scale 使用原策略；较早的 B scale NT 实验已撤回。

目标形状的 GEMM2 默认走每专家 8×384+1×512 列、252 个有效 tile、8 waves、4 个 B LDS 槽的 native FP32 输出。B 权重保留 NT 读取；A probe、alias scope 和 barrier 未删除。独立 reduce 逐 route 乘权重、舍入到 BF16、用 FP32 求和并最终输出 BF16。`EP32_G2_NATIVE=0` 可选择本树 BM80 atomic 基线。

reduce 在原 parser 中单列，不进入 FC1+FC2 BW 分母。该指标提升不能直接代表整个 fmoe 延时改善；独立 reduce 的代价必须同时报告。

## 验证与计量

最后一次有效的未修改原 bench 默认 CLI 捕获，HIP5、3 次预热、30 个样本、512 MiB 只读 flush：

| FC1 µs | FC2 µs | 逐样本合计中位数 µs | FC1+FC2 TB/s | reduce µs | GPU busy µs |
|---:|---:|---:|---:|---:|---:|
| 68.881 | 41.081 | 110.081 | 4.782974 | 24.8205 | 170.022 |

FC1 bytes 为 341592672、FC2 bytes 为 184921920、合计 526514592。5 TB/s 要求合计中位数 ≤105.3029184 µs；FC2 单独 5 TB/s 要求 ≤36.984384 µs。合计中位数由逐样本相加后计算，不使用两个独立中位数相加。

较早的同进程交替配对 native 捕获达到过 5.115716T（FC1 67.940、FC2 34.8205、合计102.921 µs），当时 G1 B scale NT 开启；默认连续 native 仍为约4.78T，差异尚未查清。配对结果不能替代默认验收，也不能据此声称已达5T。该配对 GPU busy 从 atomic 的143.961变为native的163.621 µs。

正确性记录：原 bench 实际 BM80 输入有2046 routes、1650活跃token、28专家、2240 padded排序行。三次≤2-route token重放、全部28专家隔离贡献与冻结atomic基线逐位相同；输出预置NaN后全部写入且finite。独立随机路由 E2E 使用896 global experts/EP32/local28、M4096、topk16，`checkAllclose_err=8.4126e-6`、`logits_diff=2e-6`，PASSED，并确认260次native dispatch。≥3-route的舍入/求和顺序与原BF16 atomic不同，没有宣称逐位一致。

提交前默认源码与上述有效捕获保存的哈希逐项一致。CSV遵循仓库`text/eol=lf`规则；原捕获工作树为CRLF，Git归档只归一化换行，行值不变。两种哈希保存在[records/source_manifest.json](records/source_manifest.json)。

## 未接入默认的候选

以下仅完成离线编译，未验证GPU正确性或性能。它们以独立patch归档，不在当前默认运行路径中。

| 候选 | 验证内容 | 归档 |
|---|---|---|
| G2 native FP32 store NT | 2679条ISA逐项比较，只有20条store增加NT；资源、MFMA和等待不变 | [gemm2_nt_store.patch](experiments/gemm2_nt_store.patch) |
| G2 grow-only byte scratch | 复用既有`_stage2_scratch`，对照直接缓存FP32的分配方式；未测性能 | [gemm2_grow_scratch.patch](experiments/gemm2_grow_scratch.patch) |
| G2 pingpong scratch | 仅保留最小源码候选，等前面的暂存诊断提供依据后再考虑跑 | [gemm2_pingpong_scratch.patch](experiments/gemm2_pingpong_scratch.patch) |
| G1 contiguous sorted FP8 A | 每次准备有效排序行；560条MFMA和82条buffer-to-LDS load数量不变，G1 scratch0 | [gemm1_sorted_a.patch](experiments/gemm1_sorted_a.patch) |

每个patch独立基于本分支默认源码。不要同时应用两个scratch替换patch。G1候选只接受指定bench形状；其额外准备复制不进入原BW分母，需同时查看prepare和busy耗时。

最后一轮HIP0–7均被其他GPU任务占用。HIP5空闲检查失败（gfx100、UMC19），没有启动候选GPU测试。此前一次native-only诊断在gate失败后仍被执行，已明确作废；其约3.03T不能作为回归或对照数据。

## 资料与复现

[records/g2var/default_summary.json](records/g2var/default_summary.json)保存默认结果；同目录保存正确性、配对和ISA资源摘要。[workspace_tools](workspace_tools)保存cx工具及实验阶段的文档快照；其中“未commit/push”等表述对应这些实验完成时的历史状态。本提交没有归档模型tensor、JIT cache、code object或大体积原始trace。

工具保留用户指定的工作区/环境路径。将此仓库放在`/root/workspace/workflow_34/moonshot_ep32_moe_v2/cx/aiter`，共享bench和cc工具保持原路径后，可把工具复制回cx。候选生成器会从当前默认源码重新生成完整实验副本和不可变基线，无需提交数份大kernel源码副本。

```bash
EP32_CX=/root/workspace/workflow_34/moonshot_ep32_moe_v2/cx
cp -R docs/ep32_m4096_native_v2/workspace_tools/. "$EP32_CX/"
cd "$EP32_CX"
EP32_PY=/tmp/aiter_ep32_branch_review_20261112/bin/python
"$EP32_PY" -B g2continuity/prepare.py generate
"$EP32_PY" -B g1sorted/prepare.py generate
EP32_IGNORE_RESERVED_VRAM=1 bash run_ep32_baseline.sh 5
```

每条GPU命令均先执行指定`gpu_idle.sh`及连续10秒gfx/umc空闲检查；仅按用户授权忽略保留显存。候选`run.sh`在gate通过后才安装，退出时按字节校验并恢复默认源码。使用FlyDSL0.5.0及`cc/wt_cache/flydsl050`。本次commit/push阶段只核对哈希、AST、shell语法和patch可应用性，没有重跑GPU测试。
