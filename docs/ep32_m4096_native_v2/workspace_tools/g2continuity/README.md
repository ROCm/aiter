# 默认 GEMM2 连续运行差异诊断（2026-11-13）

默认尚未达标。有效原bench默认入口为FC1 68.881 µs、FC2 41.081 µs、逐样本合计中位数110.081 µs / 4.7830T；配对曾达到5.1157T，但不能代替默认入口验收。本轮没有新的有效GPU捕获。

本轮先ListAgents/列出会话，再向已交接的会话确认其无GPU作业，只在v2/cx工作；没有读或修改v3、cc/aiter。

## 假设与最少测试次序

1. 纯转发入口：如果early import或CPU调用节奏导致差异，原bench的连续native经纯转发后应恢复约35 µs。仅作为诊断，不作为默认验收。native_only_bench.py现在使用原parser，不再错误地按奇偶拆样本。
2. 暂存方式：如果分配地址/缓存状态是原因，用既有_stage2_scratch的grow-only byte backing应改变默认连续表现。该试验只替换wrapper，不改GPU kernel。pingpong另保留一个最小生成版本，只有grow试验提供依据时才考虑跑；没有实际测过。
3. FP32写出策略：如果native destination的cache replacement造成代价，给native FP32 store加NT应降低默认FC2。保留全部等待、MFMA及数学，只变cache hint；不使用device-scope或删barrier探针。

另一会话说明HMAX只用于名字、topk不影响native分支；其暂存池也是单个buffer。其grow-only backing按(device,stream,slot)缓存uint8，再view到FP32。本树原来按(device,stream,max_sorted)直接缓存FP32。尚无实测证明这就是原因，未盲目修改默认接入。

## 当前完成程度

NT候选与原版均离线编译通过。2679条ISA指令逐项比较，恰好20条buffer_store_dwordx4增加nt，其他指令逐项一致。实际next_free_vgpr256、next_free_sgpr96、LDS161792 B、private/scratch0，均未变。isa_comparison.json保存证据。

2026-11-13 01:02 UTC在HIP5尝试默认入口，gpu_idle_for_bench.sh未通过：gfx100、UMC19、显存266337 MB。run.sh已在gate失败时显式退出，没有运行bench，也没有替换生产文件；证据在single_default_20261113_010252_hip5/gpu_idle.log。其他HIP0–7快照也均为非零gfx/umc。

此前g2var/native_only_20261113_003942在gate失败后仍被执行，已标INVALID_PERFORMANCE.md；其3.03T不能用于因果对照或回归判断。

## 复现

候选和原字节快照在source/，prepare.py install/restore会校验当前文件字节，不覆盖意外变化。run.sh在GPU空闲之后才安装，一律退出时恢复。正式bench和parser未修改。默认指标需用原CLI，3预热/30样本/512MiB flush，每个试验只跑M4096/EP32/rank0。

```bash
cd /root/workspace/workflow_34/moonshot_ep32_moe_v2/cx
# 首先只验证连续native纯转发是否仍慢；不能拿它作为达标验收。
EP32_IGNORE_RESERVED_VRAM=1 bash g2continuity/run.sh 5 single forward
# 有空闲卡后，NT先正确性、再原CLI；两条命令各自重新gate。
EP32_IGNORE_RESERVED_VRAM=1 bash g2continuity/run.sh 5 nt check
EP32_IGNORE_RESERVED_VRAM=1 bash g2continuity/run.sh 5 nt default
# 只有上述诊断需要时，单独比较grow wrapper，避免扫参。
EP32_IGNORE_RESERVED_VRAM=1 bash g2continuity/run.sh 5 grow default
```

使用指定Python、FlyDSL0.5.0和cc/wt_cache/flydsl050。用户允许忽略保留显存，仍要求gfx/umc连续10秒为0。NT正确性和性能均未验证，当前默认源码没有安装它。没有commit/push。
