# EP32/M4096：BM80 + 可变列宽 Native GEMM2

当前默认入口尚未达到5 TB/s：最后一次有效默认捕获为FC1 68.881 µs、FC2 41.081 µs、逐样本合计中位数110.081 µs / 4.7830 TB/s。配对捕获曾达到5.1157 TB/s，但不能替代默认入口验收；差异仍在诊断。default_summary.json记录当前验收状态。

当前默认fmoe使用BM80排序/GEMM1，GEMM1 B scale NT已撤回。GEMM2采用每专家8×384+1×512列的252个有效tile、8 waves、4个B LDS槽和原生MFMA FP32输出，最后由独立reduce乘route weight、逐route舍入到BF16、FP32求和并输出BF16。模型字节和正式bench未改变。

## 计量与实测

FC1 bytes341592672、FC2 bytes184921920、合计526514592。5T要求逐样本FC1+FC2之和的中位数≤105.3029184 µs；FC2单独5T要求≤36.984384 µs。原parser将独立reduce单列，不进入该BW分母。

HIP5，同进程交替，各实现3次预热、30个样本，512 MiB只读flush，均使用本树BM80 G1/SNT配置，只切换G2：

| G2 | FC1 µs | FC2 µs | 合计中位数 µs | 合计TB/s | reduce µs | GPU busy µs |
|---|---:|---:|---:|---:|---:|---:|
| BM80/BNT atomic基线 | 68.2805 | 39.7205 | 108.0010 | 4.8751 | 0 | 143.961 |
| 可变列宽native | 67.9400 | 34.8205 | 102.9210 | 5.1157 | 25.280 | 163.621 |

提升是脚本定义的FC1+FC2 BW；独立reduce增加了全流程耗时，不能把这个BW提升说成全fmoe延时改善。paired_final_summary.json保存完整捕获路径与结果。

## 正确性

check80.py从未修改的原bench实际构造输入获取BM80调用：2046 routes、1650活跃token、28专家、2240 padded sort rows。与不可变来源的BM80 atomic实现比较，3次完整≤2-route token逐位相同、全部28专家隔离贡献逐位相同；输出预置NaN，所有输出均写入且finite。check80_summary.json/check80.log保存证据。

独立op_tests/test_moe_ep.py随机路由E2E：global M4096、H3584/I3072、896 global experts、EP32/local28、topk16、SiTUv2 beta4/linear25；T1592、R1997。PASSED，checkAllclose_err8.4126e-6、logits_diff2e-6，并统计260次native dispatch，保证验收实际进入本树新路径。该测试的-e895会得到local27并走其他配置，不能当作本路径验收。

≥3-route使用逐route BF16舍入后FP32累加，与原BF16 atomic的逐加舍入/不确定顺序不同；未宣称这部分逐位相同。独立参考测试验证最终数值。

## 实现与边界

仅修改现有CSV的EP32/M4096/topk16或17两个配置：block_m80及G1/G2 kernel名字。G1保留原SiTUv2、BF16物化和FP8量化，补80行块在32行scale块中的对齐。其他模型不在本轮目标内。

G2的FP32暂存直接存[sorted row16, col16, lane64, fragment4]；reduce读reverse routes，只累加local且valid贡献。G2仍保留per-slot alias、A probe和workgroup barrier；没有使用已失效的无barrier探针。B NT使用Python整数2，未使用fx.Int32(2)。

代码在aiter/fused_moe.py、ep32_native80.py、mxmoe_gemm2_ep32_bm80var.py、mxmoe_reduce_ep32.py等。原SBM64/BNT kernel保留为其他配置的路径。EP32_G2_NATIVE=0可选择本树BM80 atomic基线；默认1选择native。

候选来源由另一个已协调会话通过消息交接，未读取v3目录。g2var_shared/package.json的SHA为c4763e23146fab22ebe6b08d5d1f465a01ab9b7339c3c43cb9bb28058722ba0c；从本树2ee6af1df基线重建var源码，SHA93bd48b7bd8e541625de88cb659351672650ebf8ea21820989792278f49b5dc4完全匹配，然后在本树独立测试。源目录和JSON保留供审阅。

## 已否决的SBM64移植

保留SBM64+tail16的初版native虽然通过逐专家/≤2-route检查，但原bench FC2 44.7805→43.121 µs、合计4.6909T，未达到目标。随后尝试一次准备ordered tile descriptor；descriptor与CPU完全一致，但A DMA存在重放漂移（错误小token880/781/180）。全VMEM等待能修复，单/双目的地址probe未稳定修复；均未进入默认路径。source_var64/和map_*.py保存这些实验源码，默认kernel目录已清理，不要无新依据重试。

常量路由权重及R1/R2混合atomic写回的失败记录见../g2hybrid/README.md；G1 split-K/stream-K失败记录见../g1stream/README.md。本轮没有大规模扫参，没有commit/push。

## 复现

```bash
cd /root/workspace/workflow_34/moonshot_ep32_moe_v2/cx
EP32_IGNORE_RESERVED_VRAM=1 bash run_ep32_baseline.sh 5
EP32_IGNORE_RESERVED_VRAM=1 bash g2var/run.sh 5 check
EP32_IGNORE_RESERVED_VRAM=1 bash g2var/run_e2e.sh 5
```

使用/tmp/aiter_ep32_branch_review_20261112/bin/python、FlyDSL0.5.0及cc/wt_cache/flydsl050。所有GPU命令先调用指定gpu_idle.sh，按已有授权忽略保留显存，仍要求gfx/umc连续10秒为0。
