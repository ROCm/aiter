# MegaMoE Tile — EP16 两节点 Stage2

Kimi-K3 A4W4 MoE，EP16 跨两台 8 卡 MI355X (gfx950)。本目录是 Stage2 的
性能复现脚本。

## 形状

hidden=3584, inter=3072, experts=896 (56/rank), topk=16, SiTUv2,
cross_node 每 token 每 rank 1 条路由。

**tune 表的 key `token` = TPR × topk**（EP16 下 1 route/token/rank），
不是 TPR 本身 —— `aiter/configs/megamoe_tile_stage2_tuned.csv` 按这个口径查。

## Stage2 的两个 kernel

| kernel | 名字前缀 | 做什么 |
|---|---|---|
| kernel1 | `megamoe_stage2_compact_*` | GEMM2 + 加权 P2P scatter 到对端 plane slot |
| kernel2 | `megamoe_k2_*` | phase1 node 内归约 → phase2 rail doorbell → phase3 跨节点归约 + unpermute |

kernel1 push 端不发 flag，kernel2 开头用一次全 rank barrier 同步；rank 内
用本地计数 + 赢家 relaxed store 发布到达标志。

## 跑一个用例

两个节点各跑一次，node_rank 分别是 0 和 1，master 是 node0：

```bash
# node0
bash scripts/megamoe_tile/run_case_relaxed.sh 0 <port> candidate <TPR> <tag>
# node1
bash scripts/megamoe_tile/run_case_relaxed.sh 1 <port> candidate <TPR> <tag>
```

结果落在 `trace_data/stage2_graph_20260909/<tag>/node<r>/summary.json`。

必须的环境变量（两端一致）：

```bash
export MEGAMOE_TWO_KERNEL=1        # 走两 kernel Stage2
export MEGAMOE_TK_EXPERT_MAJOR=1   # 目的端 expert-major 排布
export MEGAMOE_TK_ARRIVAL=1        # 到达标志协议
export MEGAMOE_TK_RAIL=1           # 跨节点 rail
export MEGAMOE_TK_COMM_QUANT_RAIL=fp8   # 可选:rail fp8 通信量化
```

`MEGAMOE_TK_QP` / `MEGAMOE_TK_CHUNK` **不要设** —— 不设时按 shape 查
`aiter/configs/megamoe_tile_stage2_tuned.csv`；设了就固定覆盖查表结果，
只在做单变量实验时用。kernel 名里的 `qp8_c64` 后缀是查表是否生效的唯一可信证据。

## 测量口径

cudagraph 计时，warmup 10、iterations 40、**取后 20 轮**，16 个 rank 池化成
320 个样本，报 min / mean / p95。跨 harness 比较**必须先对齐统计量**：
`pooled_min` 会藏住 rank 间离散度。

## 基线（本仓实测，待回填）

| 用例 (TPR=512, token=8192) | min | mean | p95 | relL2 |
|---|---|---|---|---|
| bf16 | 359.8 µs | 374.4 | 391.9 | 0.0 (逐位一致) |
| rail fp8 | **322.5 µs** | 335.1 | 352.0 | 0.026690 |

16 rank 池化 320 样本 (16 × 后 20 轮)。kernel 名
`megamoe_k2_h3584_t512_k16_qp8_c64_w4_aw1_qr8` —— `qp8_c64` 是查表生效的证据,
`qr8` 是 rail fp8 生效的证据。**summary.json 里的 `candidate_return_chunk_tokens`
和 `candidate_rail_quant_type` 是 harness CLI 层字段,不是算子实际用的值,别拿它当证据。**

## 环境前提

本仓的 `aiter/ops/flydsl/kernels/act.py` 调 `fx.max`,那是 **flydsl 0.3.2**
才有的符号。容器里若装的是 0.3.1,任何走 a4w4 fused-moe 的路径都会报
`AttributeError: module 'flydsl.expr' has no attribute 'max'`,与 megamoe_tile
无关。升级 flydsl 到 >= 0.3.2,或用 PYTHONPATH 挂一份。

## CCO 用法

host 端和 device 端都直连 MORI 公开 API：

* host —— 测试脚本建 `Communicator`（rank0 出 unique id、广播、`Communicator.init`），
  作为 `communicator=` 传给算子；teardown 先 `operator.close()`（只还算子自己的
  window/memory）再 `comm.destroy()`。和 `op_tests/multigpu_tests/bench_mega_moe.py`
  同一套。算子不传 communicator 时退回自建，单测不必自己做 rendezvous。
* device —— node 内寻址用 `mori.cco.device.flydsl` 的 `Window.lsa_ptr`，
  window 的 host 侧读写用 `window_view.py` 的 torch 视图（零拷贝，不需要
  hipMemcpy 包装，也不需要清零 kernel）。

唯一的例外是 `gda_rail.{cpp,py}`，只为两件 MORI 公开绑定表达不了的事存在：

1. **rail team** —— 它导出的每个 GDA 符号都取 `ccoTeamMode` 的默认值
   `CCO_TEAM_WORLD`（`cco_scale_out.hpp:212`），`peer` 被当作 world rank 解析；
   跨节点返回路径要按 **node index** 走 `CCO_TEAM_GDA`，没有符号够得到。
2. **推迟门铃** —— `ccoGdaOptFlagsAggregateRequests` 让 WQE 入队但不敲门铃，
   这样一整批 chunk 可以攒好再由一次 `put_value` + flush 释放。MORI 的 wrapper
   在 SDMA 路径传 optFlags，GDA 路径不传。注意这和 `at` 符号的
   `ccoGdaThreadAggregate` **不是一回事**——后者是把 warp 各 lane 合并成一次
   传输，不推迟任何东西。

只单态化了 (rail, warp) 这一个组合，因为算子只用这一个。等 MORI 的 wrapper
给 GDA 加上 team 参数和 optFlags，这两个文件就可以删掉。
