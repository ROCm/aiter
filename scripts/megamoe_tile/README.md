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

两个节点各跑一次，node_rank 分别是 0 和 1，master 是 node0；同一个网络的所有
(TPR, fixture) case 在一个进程里跑完（只建一次链，权重只生成一次）：

```bash
# node0 / node1
bash scripts/megamoe_tile/run_internode_test.sh 0 <master_ip> <port> <tag> -- \
  --network kimi_k3 --tpr-list 128,256,512,1024 --fixtures eplb,random
bash scripts/megamoe_tile/run_internode_test.sh 1 <master_ip> <port> <tag> -- <同上>
```

每个 case 依次做：fused 的 eager 输出对 MORI A4W4 参照（同一份权重）的 relL2，
graph replay 对 eager（中间改输入、换路由各验一次），然后计时。选项：

* `--timing total,breakdown`（默认两者）：`total` 是总耗时，`breakdown` 是分段，
  `none` 只验精度；
* `--paths fused,smallop`（默认两者）：fused 算子和/或小算子基线
  （quant + MORI dispatch + 调过的 fused_moe + MORI combine；DSV4 是 a8w4）；
* `--no-check` 跳过参照和 replay 检查。

最后打印一张 `[SUMMARY]` 表（每个 case 一行：relL2、fused / 小算子总耗时、加速比、
fused 的 stage1 / stage2 分段）。

日志在 `trace_data/internode/<tag>/node<r>.log`，结果（rank 0）在
`trace_data/internode/<tag>/results.json`。

配置不走环境变量：算子里写死的是实测最优配置，按 shape 变的参数查两张表
（`aiter/configs/megamoe_tile_stage1_tuned.csv`：tile_group / split_local /
fanout_shards；`megamoe_tile_stage2_tuned.csv`：return_chunk_tokens / gemm2_bn /
gemm2_cu），查不到时按 `stage1_tune.py` / `stage2_tune.py` 里的规则推。扫参时把
`AITER_CONFIG_MEGAMOE_TILE_STAGE1` / `AITER_CONFIG_MEGAMOE_TILE_STAGE2` 指向自己的表。
跨节点 rail 段的 fp8 通信量化是构造参数 `comm_quant_rail="fp8"`（测试默认开，
`--no-rail-fp8` 关）。kernel 名里的 `_tg` / `_fos` / `_slg` / `_qr8` 等后缀是实际配置
的唯一可信证据。

## 测量口径

和 `op_tests/multigpu_tests/bench_mega_moe.py` 一样：一个 graph 里是一次完整
forward，warmup 10 次后计时 40 次，每次 replay 后 synchronize、用 host 墙钟计时。
总耗时报每个 rank 的 min / median 再对 16 个 rank 取平均。分段在计时之后另跑
torch.profiler：20 次 replay，取最后 5 次，每行是每个 rank 的最小值再取平均；
profiler 偶尔丢事件，所以它不进总耗时。跨 harness 比较**必须先对齐统计量**：
`pooled_min` 会藏住 rank 间离散度。

下面的基线是旧口径（profiler span、后 20 轮池化），墙钟要高 ~10 µs，不能直接比。

## 基线（本仓实测，待回填）

| 用例 (TPR=512, token=8192) | min | mean | p95 | relL2 |
|---|---|---|---|---|
| bf16 | 359.8 µs | 374.4 | 391.9 | 0.0 (逐位一致) |
| rail fp8 | **322.5 µs** | 335.1 | 352.0 | 0.026690 |

16 rank 池化 320 样本 (16 × 后 20 轮)。kernel 名
`megamoe_k2_h3584_t512_k16_qp8_c64_w4_aw1_qr8` —— `qp8_c64` 是查表生效的证据,
`qr8` 是 rail fp8 生效的证据。

## 环境前提

不占用 GPU 的 host 回归可单独运行：

```bash
python op_tests/test_megamoe_tile_host.py -v
```

检查计时统计及 162 组 arena 配置的区域边界和 RDMA 注册前缀；布局检查需要
PyTorch，不需要初始化分布式通信或分配 GPU 内存。

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

## rail fp8 精度(6 个可跑的 token 点,每点配 bf16 对照)

| token | TPR | bf16 relL2 | fp8 relL2 | bf16 min | fp8 min | fp8 省 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 0.000000 | 0.026709 | 148.8 µs | 146.1 µs | 1.8% |
| 1024 | 64 | 0.000000 | 0.026759 | 151.9 | 151.4 | 0.3% |
| 2048 | 128 | 0.000000 | 0.026698 | 203.3 | 194.3 | 4.4% |
| 4096 | 256 | 0.000000 | 0.026682 | 249.8 | 231.2 | 7.4% |
| 8192 | 512 | 0.000000 | 0.026690 | 361.6 | 322.8 | 10.7% |
| 16384 | 1024 | 0.000000 | 0.026672 | 603.7 | 533.0 | 11.7% |

bf16 六个点全部逐位一致,所以偏差 100% 来自 fp8。fp8 误差跨 32 倍 token 只摆动
0.33% 且不单调 —— 和 1x32 e8m0 分块量化的性质一致(误差只取决于块内动态范围,
与 token 数无关),可以按常数 0.0267 计入误差预算。

量级也对得上:e4m3 相对步长 2^-4,均匀量化 RMS ≈ 步长/√12 ≈ 1.8%/元素;
跨节点归约里**两个**项都量化过,√2 × 1.8% = 2.55%,实测 2.67%。

**这个 relL2 的口径是 candidate vs MORI bf16 基线**,隔离了 fp8 这一个变量,
不等于端到端精度。`test_wide_ep_moe.py`(13 shape / 阈值 0.15)走的是上游
MORI dispatch + fused_moe,没有接本算子的钩子,所以两者无法直接叠加测量。

13 个 Kimi shape 里当时只测了这 6 个。token 32768 曾撞 RDMA MR 注册上限(之后改成只注册
RDMA 前缀);token 131072 超 `stage1_abi.py` 的 `MAX_FUSED_TOKENS_PER_RANK = 4096`。
