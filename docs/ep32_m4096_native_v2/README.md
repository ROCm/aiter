# EP32 M4096 BM80 native GEMM2：v2 工作树记录

本分支保存 Kimi-K3 A8W4、H3584/I3072、EP32/local28、M4096 的当前实现和待验证优化资料。原 `amd_moe_bench_share/bench_moe.py` 及 parser 未修改。**本工作树未修改原bench默认CLI的一次30样本捕获已达到5.050936 TB/s。** 当前仅有一次成功的输出NT默认捕获；后续复测停在GPU空闲检查，未启动bench。

## 默认实现

现有 CSV 只更新 M4096/topk16、17 两行的 block_m 和 kernel 名称。GEMM1 支持 BM80 排序，并处理 80 行块跨 32 行 scale 块的对齐，保持原 SiTUv2、BF16 materialization 和 MXFP8 量化。GEMM1 B scale 使用原策略；较早的 B scale NT 实验已撤回。

目标形状的 GEMM2 默认走每专家 8×384+1×512 列、252 个有效 tile、8 waves、4 个 B LDS 槽的 native FP32 输出。B 权重保留 NT 读取，native FP32写出新增NT提示；A probe、alias scope 和 barrier 未删除。独立 reduce 逐 route 乘权重、舍入到 BF16、用 FP32 求和并最终输出 BF16。`EP32_G2_NATIVE=0` 可选择本树 BM80 atomic 基线。

reduce 在原 parser 中单列，不进入 FC1+FC2 BW 分母。该指标提升不能直接代表整个 fmoe 延时改善；独立 reduce 的代价必须同时报告。

## 验证与计量

最新有效的未修改原bench默认CLI捕获，HIP1、3次预热、30个样本、512MiB只读flush。该捕获通过临时安装NT源码运行，退出后恢复；随后默认代码安装同一源码，哈希逐项一致：

| FC1 µs | FC2 µs | 逐样本合计中位数 µs | FC1+FC2 TB/s | reduce µs | GPU busy µs |
|---:|---:|---:|---:|---:|---:|
| 68.401 | 35.6205 | 104.241 | 5.050936 | 26.54 | 169.482 |

FC1 bytes 为 341592672、FC2 bytes 为 184921920、合计 526514592。5 TB/s 要求合计中位数 ≤105.3029184 µs；FC2 单独 5 TB/s 要求 ≤36.984384 µs。合计中位数由逐样本相加后计算，不使用两个独立中位数相加。

较早的同进程交替配对 native 捕获达到过 5.115716T（FC1 67.940、FC2 34.8205、合计102.921 µs），当时 G1 B scale NT 开启；输出NT加入前默认连续native曾为4.783T；本轮未加输出NT的HIP1独立默认捕获为4.968T。它们均来自不同捕获，不据此推导NT的稳定收益。配对结果不能替代默认验收；本次5T结论来自上述原CLI输出NT捕获。该历史配对GPU busy从atomic的143.961变为native的163.621 µs（增加19.66 µs，13.7%）；首末kernel跨度从304.623变为324.483 µs（增加19.86 µs，6.5%）。不是CPU入口到返回的完整延时测量。输出NT新版本的同次atomic对照未启动，不能将历史净增量当成其精确值。

正确性记录：原 bench 实际 BM80 输入有2046 routes、1650活跃token、28专家、2240 padded排序行。三次≤2-route token重放、全部28专家隔离贡献与冻结atomic基线逐位相同；输出预置NaN后全部写入且finite。独立随机路由 E2E 使用896 global experts/EP32/local28、M4096、topk16，`checkAllclose_err=8.4126e-6`、`logits_diff=2e-6`，PASSED，并确认260次native dispatch。≥3-route的舍入/求和顺序与原BF16 atomic不同，没有宣称逐位一致。

NT实际bench输入验证通过全部28专家隔离贡献、≤2-route token比较和三次默认dispatch NaN/finite重放，见records/g2continuity/nt_check_summary.json。当前默认源码与上述有效NT捕获保存的哈希逐项一致。CSV遵循仓库`text/eol=lf`规则；原捕获工作树为CRLF，Git归档只归一化换行，行值不变。两种哈希保存在[records/source_manifest.json](records/source_manifest.json)。

## 候选与验证状态

G2输出NT已完成上述GPU验证并接入默认；其他候选只保留离线/源码资料。patch及workspace_tools保留输出NT接入前（fa961eb62）的实验基线，不能把历史NT patch再次应用到当前默认。

| 候选 | 验证内容 | 归档 |
|---|---|---|
| G2 native FP32 store NT | 已接入；2679条ISA只有20条store增加NT，资源/MFMA/等待不变；GPU正确性及原CLI5.050936T通过 | [gemm2_nt_store.patch](experiments/gemm2_nt_store.patch) |
| G2 grow-only byte scratch | 复用既有`_stage2_scratch`，对照直接缓存FP32的分配方式；未测性能 | [gemm2_grow_scratch.patch](experiments/gemm2_grow_scratch.patch) |
| G2 pingpong scratch | 仅保留最小源码候选，等前面的暂存诊断提供依据后再考虑跑 | [gemm2_pingpong_scratch.patch](experiments/gemm2_pingpong_scratch.patch) |
| G1 contiguous sorted FP8 A | 每次准备有效排序行；560条MFMA和82条buffer-to-LDS load数量不变，G1 scratch0 | [gemm1_sorted_a.patch](experiments/gemm1_sorted_a.patch) |

每个patch基于输出NT接入前的代码；NT patch目前已应用，其他patch保留为候选。不要同时应用两个scratch替换patch。G1候选只接受指定bench形状；其额外准备复制不进入原BW分母，需同时查看prepare和busy耗时。

本轮在HIP1通过空闲检查后完成NT实际输入正确性和原CLI性能捕获。后续默认复测、当前NT/atomic开销对照停在空闲检查，未启动bench。此前一次native-only诊断在gate失败后仍被执行，已明确作废；其约3.03T不能作为回归或对照数据。

## 资料与复现

[records/g2var/default_summary.json](records/g2var/default_summary.json)保存默认结果；同目录保存正确性、配对和ISA资源摘要。[workspace_tools](workspace_tools)保存cx工具及实验阶段的文档快照；其中“未commit/push”等表述对应这些实验完成时的历史状态。本提交没有归档模型tensor、JIT cache、code object或大体积原始trace。

工具保留用户指定的工作区/环境路径。将此仓库放在`/root/workspace/workflow_34/moonshot_ep32_moe_v2/cx/aiter`，共享bench和cc工具保持原路径后，可把工具复制回cx。当前默认复现只需run_ep32_baseline.sh和gpu_idle_for_bench.sh。要回放旧候选生成器，先使用fa961eb62的独立实验checkout；这些历史生成器的锚点和基线哈希对应输出NT接入之前，无需提交数份大kernel源码副本。

```bash
EP32_CX=/root/workspace/workflow_34/moonshot_ep32_moe_v2/cx
cp -R docs/ep32_m4096_native_v2/workspace_tools/. "$EP32_CX/"
cd "$EP32_CX"
EP32_PY=/tmp/aiter_ep32_branch_review_20261112/bin/python
EP32_IGNORE_RESERVED_VRAM=1 bash run_ep32_baseline.sh 5
```

每条GPU命令均先执行指定`gpu_idle.sh`及连续10秒gfx/umc空闲检查；仅按用户授权忽略保留显存。候选`run.sh`在gate通过后才安装，退出时按字节校验并恢复默认源码。使用FlyDSL0.5.0及`cc/wt_cache/flydsl050`。本轮只测试指定EP32/M4096点，没有扫参。发布前检查已验证源码哈希、AST/JSON/shell语法和diff；GPU实际输入验证与原CLI捕获已完成，安装后另一次复测未执行。
