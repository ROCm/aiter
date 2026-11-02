# MegaMoEInternode — EP16 两节点融合 MoE

A4W4 MoE，EP16 跨两台 8 卡 MI355X (gfx950)。算子是
`aiter.ops.flydsl.kernels.megamoe_internode.MegaMoEInternode`，测试是
`op_tests/multigpu_tests/test_megamoe_internode.py`。

## 接口

和 `MegaMoEV2`（`aiter.ops.flydsl.kernels.mega_moe`）一致：

```python
from aiter.ops.flydsl.kernels.megamoe_internode import MegaMoEInternode

moe = MegaMoEInternode(rank=rank, world_size=16, model_dim=H, inter_dim=I,
                       experts=E, topk=K, quant="a4w4", w1=w1, w1_scale=w1s,
                       w2=w2, w2_scale=w2s, max_tok_per_rank=T,
                       swiglu_limit=0.0, activation="silu", comm_quant="fp8")
out = moe(x_bf16, wts, topk_ids)                 # = moe.forward(...)
x_q, scales = moe.quantize(x_bf16)               # MXFP4 + E8M0
out = moe.forward_prequant(x_q, scales, wts, topk_ids)
moe.close()
```

与 MegaMoEV2 的差别：

* 只支持 `quant="a4w4"`、`world_size=16`（2 节点 × 8 卡）；
* 每次 forward 的行数必须等于 `max_tok_per_rank`（上限 16384）；pad 行的 topk id 填 `-1`
  即可，`-1`（及越界 id）的 slot 不 dispatch、不算、combine 按 0 处理；
* 多出四个关键字参数：`activation`（`silu` / `situv2`）、`comm_quant`
  （跨节点 rail 段的载荷格式，`fp8` 默认 / `none` 即 bf16）、`communicator`
  （调用方建的 `mori.cco` Communicator，不传则自建）、
  `max_routes_per_token_per_rank`（一个 token 落到同一 rank 的路由上限，
  决定 expert tile arena 的大小，默认 topk，调用方保证不越界）；
* `mega_scheme` / `fanout_masks` 只为接口兼容而接收，跨节点 dispatch 只有一种方案；
* `close()` 或 `with` 释放 CCO window。

forward 是同一条流上的 4 次发射，generation/parity 在 device 上推出，可以整体
捕获进 CUDA graph 反复 replay。

| kernel | 名字前缀 | 做什么 |
|---|---|---|
| 输入量化 | `per_1x32_mx_quant_fp4_*_railrec_*` | MXFP4 量化，连同 ids/weights 直接写成窗口里的 rail record |
| stage1 | `megamoe_stage1_*` | InterNodeV1 dispatch 到 expert tile + GMM1 + 激活 + A4 量化 |
| stage2 kernel1 | `megamoe_k1_*` | GEMM2 + 加权 push 到 node 内代理 GPU 的 plane-slot inbox |
| stage2 kernel2 | `megamoe_k2_*` | node 内归约 → rail 发往对端节点 → 跨节点归约 + unpermute |

kernel1 每写完一个 (token, topk slot) 行，由最后一个写它的 CTA 发布到达标志
（本地计数 + 赢家 relaxed store）；kernel2 按 chunk 等这些标志再归约。

**tune 表的 key `token` = TPR × topk**（GEMM 行数），不是 TPR 本身。

## 代码结构

`aiter/ops/flydsl/kernels/megamoe_internode/`：

| 文件 | 内容 |
|---|---|
| `mega_moe_internode.py` | host 侧 `MegaMoEInternode`：window 分配、编译、4 次发射 |
| `mega_moe_internode_config.py` | stage1 / stage2 两张 tune 表的加载与查不到时的规则 |
| `layout.py` | 两个 stage 共用的注册 window 布局（stage1 arena + 嵌入的 stage2 区） |
| `quant.py` | 输入量化 kernel（写成 rail record） |
| `stage1.py` | stage1 kernel：dispatch + GMM1 |
| `stage2.py` | stage2 kernel1（GEMM2 + push）与 kernel2（node 归约 + rail + combine） |
| `gemm.py` | GMM1 / GMM2 的 kernel 体 |
| `communication.py` + `gda_rail.cpp` | device 侧通信原语与 rail GDA extern |
| `common.py` | 与 `mxfp4_gemm_common` 不同的几个 GEMM helper |

mtpr > 4096 时（h1 / k1 push 平面 / k2 inbox 超过 2 GiB）GMM1、k1、k2 改为按行或按 plane
建 64 位基址，`node_accumulator` / `remote_partial_rx` 按 rail 行宽分配以保证 NIC 注册前缀
< 1 GiB；≤ 4096 的布局和 ISA 与之前逐字节一致。DSV4 在 16384 下用 bf16 rail 会超过 1 GiB，
构造时直接拒绝。

## 跑测试

入口就是测试文件本身。两个节点跑同一条命令，只有 `--node-rank` 不同（0 / 1），
`--master-addr` 填 node0 的地址。用普通 `python3` 启动时它是 launcher：设置
传输层环境，然后按 `--network` 的顺序逐个拉起 torchrun（第 i 个网络用
`--master-port + i`）。同一个网络的所有 (TPR, fixture) case 在一个进程里跑完
（只建一次链，权重只生成一次）：

```bash
# node0；node1 换成 --node-rank 1
python3 op_tests/multigpu_tests/test_megamoe_internode.py \
  --node-rank 0 --master-addr <node0_ip> --master-port 29500 --tag <tag> \
  --network kimi_k3,dsv4 --tpr-list 16384,4096,1024,512,128,64,32 --fixtures random,eplb
```

launcher 参数：`--timeout`（每个网络的秒数，默认 3600，超时整组 SIGTERM 再 SIGKILL）、
`--socket-ifname`（Gloo / MORI bootstrap 网卡，默认 `enp193s0f1np1`）、
`--torchrun-wrap`（torchrun `--no-python` 包装脚本，如只在一个 rank 上跑
`rocprofv3 --att`）、`--max-idle-vram-gib`（preflight 认为空闲的显存上限，默认 2）。
MORI 的 NIC 相关变量（`MORI_DEVICE_NIC` 等）若已在 shell 里设置则沿用。

每个 case 依次做：fused 的 eager 输出对 MORI A4W4 参照（同一份权重）的 relL2，
graph replay 对 eager（中间改输入、换路由各验一次），然后计时。选项：

* `--timing total,breakdown`（默认两者）：`total` 是总耗时，`breakdown` 是 fused 的分段
  （小算子不做分段，见下面「测量口径」），`none` 只验精度；
* `--paths fused,smallop`（默认两者）：fused 算子和/或小算子基线
  （quant + MORI dispatch + 调过的 fused_moe + MORI combine；DSV4 是 a8w4）；
* `--no-check` 跳过参照和 replay 检查。

最后打印一张 `[SUMMARY]` 表（每个 case 一行：relL2、fused / 小算子总耗时、加速比、
fused 的 stage1 / stage2 分段）。

输出同时打到终端和 `trace_data/internode/<tag>/node<r>.log`，退出码写到
`node<r>.status`（任一网络失败即非 0；tag 已有 status 时拒绝启动）。结果（rank 0）在
`trace_data/internode/<tag>/results.json`；多个网络时在 `<tag>/<network>/results.json`。

配置不走环境变量：算子里写死的是实测最优配置，按 shape 变的参数查两张表
（`aiter/configs/megamoe_internode_stage1_tuned.csv`：tile_group / split_local /
fanout_shards；`megamoe_internode_stage2_tuned.csv`：return_chunk_tokens / gemm2_bn /
gemm2_cu），查不到时按 `mega_moe_internode_config.py` 里的规则推。扫参时把
`AITER_CONFIG_MEGAMOE_INTERNODE_STAGE1` / `AITER_CONFIG_MEGAMOE_INTERNODE_STAGE2` 指向自己的表。
跨节点 rail 段的 fp8 通信量化是构造参数 `comm_quant`（测试用 `--comm-quant fp8|none`，
默认 fp8）。kernel 名里的 `_tg` / `_fos` / `_slg` / `_qr8` 等后缀是实际配置的唯一可信证据。

## 测量口径

和 `op_tests/multigpu_tests/bench_mega_moe.py` 一样：一个 graph 里是一次完整
forward，warmup 10 次后计时 40 次，每次 replay 后 synchronize、用 host 墙钟计时。
总耗时报每个 rank 的 min / median 再对 16 个 rank 取平均。分段在计时之后另跑
torch.profiler：20 次 replay，取最后 5 次，每行是每个 rank 的最小值再取平均；
profiler 偶尔丢事件，所以它不进总耗时。只对 fused 做分段：小算子的 graph 在
torch.profiler 下 replay 偶发在 HIP 内死锁（roctracer 的完成回调和发射线程互等），
16 个 rank 一起挂住，所以小算子只报总耗时。跨 harness 比较**必须先对齐统计量**：
`pooled_min` 会藏住 rank 间离散度。

下面的基线是旧口径（profiler span、后 20 轮池化），墙钟要高 ~10 µs，不能直接比。

## 基线（本仓实测，待回填）

| 用例 (TPR=512, token=8192) | min | mean | p95 | relL2 |
|---|---|---|---|---|
| bf16 | 359.8 µs | 374.4 | 391.9 | 0.0 (逐位一致) |
| rail fp8 | **322.5 µs** | 335.1 | 352.0 | 0.026690 |

16 rank 池化 320 样本 (16 × 后 20 轮)。当时 kernel 名
`megamoe_k2_h3584_t512_k16_qp8_c64_w4_aw1_qr8` —— `qp8_c64` 是查表生效的证据,
`qr8` 是 rail fp8 生效的证据。

## 环境前提

aiter main 需要 **flydsl >= 0.4.1**（验证用的是 0.4.1 + triton 3.7）。

## CCO 用法

host 端和 device 端都直连 MORI 公开 API：

* host —— 测试脚本建 `Communicator`（rank0 出 unique id、广播、`Communicator.init`），
  作为 `communicator=` 传给算子；teardown 先 `operator.close()`（只还算子自己的
  window/memory）再 `comm.destroy()`。和 `op_tests/multigpu_tests/bench_mega_moe.py`
  同一套。算子不传 communicator 时退回自建，单测不必自己做 rendezvous。
* device —— node 内寻址用 `mori.cco.device.flydsl` 的 `Window.lsa_ptr`；
  host 侧清零 window、写 rail record 都是 `__cuda_array_interface__` 包出来的
  torch 视图（零拷贝，不需要 hipMemcpy 包装，也不需要清零 kernel）。

唯一的例外是 `gda_rail.cpp`（Python 侧在 `communication.py`），只为两件 MORI 公开绑定表达不了的事存在：

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
RDMA 前缀);当时 token 131072 超出 4096 的 TPR 上限,现在上限是 16384。
