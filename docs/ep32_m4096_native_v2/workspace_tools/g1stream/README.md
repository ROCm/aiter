# EP32/M4096 FC1 优化实验记录

此页记录2026-11-12的BM64历史实验，下面关于“默认源码未变”的表述只对应当时状态。2026-11-13当前默认是BM80 G1 + native G2，尚未默认达标，见../g2var/default_summary.json。此页失败结构仍不应重复。

本轮目标：不改 amd_moe_bench_share 的 bench/parse，用 fmoe 内部实现降低 FC1+FC2 GPU kernel 合计耗时，争取 5 TB/s。结论：本轮 FC1 候选均没有稳定收益，未接入默认入口；5 TB/s 未达到。默认 FC1 与不可变 baseline.py 逐字节相同，FC2 继续保留此前已验证的 B-LDS DMA NT。没有修改其他 kernel、没有 commit/push。

## 指标口径

原 parser 使用逐样本 FC1_us+FC2_us 的中位数，合计模型字节 526514592，5 TB/s 门槛为 105.3029184 µs；FC2 字节 184921920。排序、copy、清零、初始化等未匹配 FC1/FC2 名称的 kernel 归入 prepare/reduce，不进入合计 BW 分母。它们可能间接改变缓存或输入布局，本轮按用户最新指示没有修改这些部分。parser 当前还要求每个样本恰好一个 FC1 和一个 FC2 kernel；跨两阶段融合不能直接沿用当前统计规则。

本轮旧默认入口复测：HIP5，FC1 68.9805 µs、FC2 44.2205 µs、合计112.9615 µs / 4.6610 TB/s，目录 results/ep32_m4096_20261112_175729_hip5。此前 HIP4 的默认入口为68.221/40.740/109.0815 µs / 4.8268 TB/s。两次源码与配置哈希相同，但卡号、捕获时间不同，不能归因于显存保留。用户取消了显存占用因果对照，未执行该对照。另一会话 v3 的未NT FC2结果不作为本工作树NT版的对照。

## 有效性能对照

均为指定 amd bench，EP32/M4096/rank0，prequantized MXFP8，512 MiB read-only flush，同进程交替 factory，每个实现3次预热、30个样本。只在行内比较，不能用跨捕获的绝对时间推导收益。

| 候选 | 同次原版FC1 µs | 候选FC1 µs | 同次原版合计TB/s | 候选合计TB/s | 捕获 |
|---|---:|---:|---:|---:|---|
| 完整K滚动循环 + 明确DMA等待，无归并 | 69.161 | 69.261 | 4.6283 | 4.6339 | amd_pair_20261112_183005_hip5 |
| 832 WG，尾部K9+5，128-bit device-scope partial | 69.161 | 73.2205 | 4.6283 | 4.5303 | 同上 |
| 768 WG，辅助组在CU内处理两段后缀 | 69.441 | 75.241 | 4.6776 | 4.4893 | amd_pair_20261112_184435_hip5 |
| 完整K，A-tail volatile probe + B scale提前预取 | 68.8405 | 69.5605 | 4.6976 | 4.6495 | amd_pair_20261112_190924_hip5 |

有效候选均通过实际 dump 的6,285,312个实效FP8输出元素与全部scale逐位比较，3次重放无差异。分摊版本的arrival状态全部回收为零。仅验证指定路由，不推及其他通用场景。

## 实现与诊断

原版672个有效tile，每XCD84个。保留前64个完整tile，剩余20个tile的K256单元拆为9+5。832 WG版本将后缀作为独立WG派发；768 WG版本给每XCD12个辅助WG，其中8个处理两个后缀、4个处理一个。所有部分和保留FP32，在求和后执行SiTUv2、BF16物化、FP8量化。

每tile一个128 B对齐状态槽。CAS先到者写partial并发布ready，后到者读并求和、清状态。先到者不等待未派发WG，避免循环等待。128-bit buffer copy使用Python整数cache policy 16，gfx950 ISA确认load/store都有sc1，即device scope；保留显式完成与发布顺序。该接口aux=4不会产生device scope，不据参数名称猜测语义。

资源：832-WG向量版163 VGPR，768-WG resident版168 VGPR，均零scratch、51712 B LDS。早期resident版本曾因为跨段保活地址产生100 B/thread scratch；每段opaque thread-id重算地址后消除spill，仍未获得性能收益。原版160 VGPR、零scratch。

新滚动K控制的早期版本缺少A DMA明确完成等待。K为编译期常量时首次出现159190个payload差异；三次诊断重放均失败且错误元素变化。单独阻止K常量传播仍失败（351201个payload差异）。根因是workgroup barrier本身不足以保证异步A DMA完成。增加vmcnt(0)或每wave最后A-tail DMA对应的volatile LDS读取后，完整K控制三次重放逐位一致。无等待控制与初始无等待scalar-partial版本均已废弃，不能作为可信优化基线。

修正后完整K循环与原版耗时接近，主要剩余开销来自partial交换和分摊调度。B scale前移虽然正确，最终仍未获益。不会基于错误结果的无同步探针宣称提速。

## 文件、复现与清理

source/ 保存本轮9个kernel源码快照，sha256.json保存哈希；不留在默认aiter目录。baseline.py是原版FC1不可变快照，sha256=7161b32b299da6fd07f4ef8d0416338c82016b67f33683597af3c1bb7a109131。results_summary.json聚合4次捕获，原始数据在各amd_pair目录。ISA保存在各isa_*目录。

以下仅用于明确要求回放时，已经失败的方案不要重复试。run.sh自动物化源码、GPU检查，并在退出时仅删除未修改的物化文件。例：

```bash
cd /root/workspace/workflow_34/moonshot_ep32_moe_v2/cx
EP32_G1_CASE=tail9x5_resident bash g1stream/run.sh 5 check
EP32_G1_CASE=tail9x5_resident bash g1stream/run.sh 5 pair
```

按用户已授权忽略保留显存时，加EP32_IGNORE_RESERVED_VRAM=1；仍先运行指定gpu_idle.sh，再要求gfx/umc连续10秒为0。默认严格执行原显存门槛。用户最新只关注FC1/FC2，没有修改正式bench文件、排序/copy/初始化等kernel，也没有做显存因果测试。
