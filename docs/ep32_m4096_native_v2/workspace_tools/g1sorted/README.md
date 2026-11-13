# GEMM1 连续排序激活候选（2026-11-13，GPU待验证）

目标只针对未修改的amd bench：M4096、EP32/rank0、H3584/I3072、local28、topk16、BM80。候选已生成和离线编译；尚未做GPU正确性/性能检查，也未接入默认入口。

旧分析在GEMM1测到TCP in-flight request上限，移除A重读的非法probe约省5.4 µs；split-K/滚动K、resident partial、提前scale预取已失败，不再重复。此候选先把FP8代码按m_indices复制为连续排序行，GEMM1改成连续行地址读取，保留原已排序的E8M0 scale。它验证访问布局是否有收益，不预先宣称减少了A DMA次数。

## 实现与边界

prepare kernel只处理cumsum[0]内的填充行；每16行一个tile，非法token行写零，未填充容量不读不写。每线程复制14×16 B，2240排序行共写8028160 B。没有在CPU读路由/计数；排序准备的复制每次重新执行，没有缓存输出。

GEMM1以原BF16 materialization、SiTUv2和MXFP8量化精度执行；只变A代码的地址及硬件buffer bound。所有560条MFMA、82条buffer-to-LDS load保持数量相同。该数值候选必须通过冻结BM80原kernel的FP8 payload/scale逐位比较，再进正式bench。

source/保存未连接的准备kernel、wrapper、冻结原GEMM1和修改版。run.sh只在gate通过后临时替换G1，并退出时恢复；检查工具使用原bench实际输入，比较独立sorted-copy oracle、所有2240写入行及全部写入scale，重复3次，输出预置FP8 NaN。正式bench、CSV、G2均不改。实验wrapper拒绝其他形状，避免误跑旧随机路由E2E并把未覆盖路径当验证。

## 离线证据

| 项目 | 原BM80 G1 | 连续激活G1 | 准备kernel |
|---|---:|---:|---:|
| ISA指令 | 3642 | 3625 | 231 |
| next_free VGPR | 152 | 160 | 28 |
| next_free SGPR | 96 | 96 | 24 |
| LDS B | 51712 | 51712 | 0 |
| private/scratch B | 0 | 0 | 0 |
| MFMA | 560 | 560 | 0 |
| ds_bpermute | 4 | 0 | 0 |

3 WG的LDS为155136 B，小于gfx950的163840 B；3 wave/SIMD所需160×3=480 VGPR也在512范围内。实际occupancy和性能仍需实测，资源数字不作为提速结论。isa_resources.json、offline_*.log及build_*/保留证据。

准备复制不进入原脚本FC1+FC2分母，但会增加prepare耗时；正式结果需同时报告busy_us。GEMM1直接BW是否提高、全fmoe是否加速均未知。

## 复现

```bash
cd /root/workspace/workflow_34/moonshot_ep32_moe_v2/cx
EP32_IGNORE_RESERVED_VRAM=1 bash g1sorted/run.sh 5 check
EP32_IGNORE_RESERVED_VRAM=1 bash g1sorted/run.sh 5 default
```

每条命令重新调用指定gpu_idle.sh及10秒gfx/umc空闲检查；使用指定Python、FlyDSL0.5.0及cc/wt_cache/flydsl050。若gate失败，候选不会安装，也不会启动GPU测试。没有扫参、commit、push。
