# 直接把 EP dispatch+GEMM1 抄到 TP 会错在哪

日期：2026-09-18

**一句话：可以原样跑 EP 的 fused stage1，速度应对齐 `stage1`。一旦按 TP 语义改发送（一个 token 打 8 卡）或改专家布局（每卡 384 个 expert），接口和指令量都对不上，不能当真实 TP 融合。**

---

## 这次抄了什么

不改 `mega_moe_stage1.py` / `dispatch.py`。另开：

- `tp_mega_moe_stage1.py`：stage1 整核（ticket、launch_ready、planner、`num_dispatch_cu`、`build_fused_gemm1`、work pool）
- `tp_dispatch_broadcast.py`：fixed-slot payload 的 naive TP 改法——槽位仍在 owner 上分配，token/scale/header **写到全部 npes**

Bench `--tp` 多两列（CUDA Graph，和其它 TP kernel 同一套 capture）：

| 列 | 实际跑的 | 算得对不对 |
|---|---|---|
| `ep_s1_k` | `mega._run_fused_stage1` | 对，就是 EP stage1 |
| `ep_bcast_k` | 同上融合核，payload 广播到 8 卡 | 故意可以错 |
| `chunk_fused_k` | 现有 TP chunk AG + token-id GEMM1 | overlap 上限 |

`ep_s1_k` 应对齐 `[RESULT] stage1=...`。对不齐就是 capture/stream 问题。`ep_bcast_k` 多出来的就是「每个 token 打 8 卡」的 XGMI。

---

## EP 有、TP 提供不了的接口

EP stage1 假定 **expert 被 rank 切开**：`dest = global_expert // epr`，本卡只算 `epr=48` 个 expert，接收布局就是 GEMM1 的 A。

| EP 接口 | TP 实际有什么 |
|---|---|
| `p2p_rx[dest]` 按 **expert-major 固定槽**（`local_expert * cap + offset`） | all-gather 是 **rank-major 密行** `rank * m_local + t` |
| `srcmap` / `sorted_expert_ids` / `tile_row_base` 核内 grouping | TP 在 host 做 `moe_sorting`，核里没有 pair_order / hist / count_matrix |
| `payload_ready[local_expert]`，consumer 等 expert e 到齐再算 tile e | TP 没有「本卡 expert e 的 owner 发送完成」；要等的是 AG 里那些行 |
| `running[e]` 跨 rank atomic 占槽 | TP 没有 per-expert 槽；行地址是 dense AG |
| `combine` 按 srcmap 把 topk 份加权送回 token owner | TP 要的是 **reduce-scatter**（每卡一段 N，再合成 H） |
| `experts_per_rank=48`，`expert_offset=rank*48` 索引本卡 W1 | TP 每卡 **384 个 expert**，W1 是 `N/8` shard |
| fixed-slot：`epr <= 64`，`cap` 按 `npes * mtpr` 对齐 tile_m | `epr=384` 直接不能走 fixed-slot |
| dispatch CU：`_scale_dispatch_cu(cu, epr)`，epr=48 时 ×1 | 若假装 epr=384，expert_waves=6，CU 会被 cap 到 224，和 EP 的 208 不是同一套 |

核里 `ATileLoader` 按 `tile_row_base` **连续**读 A。那是 dispatch 排好的 expert 块。TP 的 `rx` 是 AG 密行，同一套 loader 读到的不是这个 tile 的 token（所以这次允许算错）。TP 正确路径要用 `TokenATileLoader` 按 `sorted_ids` 间接 gather，比 EP 多一轮 id→row。

---

## TP 会比 EP 多的指令 / 流量

EP：一个 `(token, topk slot)` 只发给 **1 个 owner rank**（`dest = expert // 48`）。topk=6 时，一个 token 最多 6 次 payload（去重 dest 后更少）。

TP 要在 **每一卡** 上算 expert0，拥有 expert0 token 的卡必须把该行送到 **全部 8 卡**，不是 1 卡。

其它多出来的：

1. **发送次数**：naive 广播把每次 payload store 乘 `npes=8`（token 7168B + scale 224B + header）。EP 一次 XGMI write 变成 8 次。owner 的 atomic 占槽仍只做一次，否则 finalize 会挂。
2. **发送内容**：EP 只发「路由到本卡 expert」的行。TP AG 把 **所有本地 token** 发给所有人，包括对 expert0 无用的行。
3. **GEMM M 维 padding**：均匀 topk=6、64 token/卡时两边 FLOPs 接近（EP 约 48 expert × 32 pad，N=6144；TP 约 384 expert × 32 pad，N=768），但 TP 多 8× M-tile、少 8× N-tile，L2 / 权重 cache 行为不同。
4. **A 的 gather**：EP 连续 load；TP 正确实现是 `sorted_ids` 间接 load（更散、更难 async copy）。
5. **Host**：TP 还要 NCCL AG `ids/weights` + `moe_sorting`。EP 这些在核内。`ep_s1_k` 没把这段算进去，和 EP `stage1` 比才公平。
6. **结尾**：EP combine；TP reduce-scatter。这次只抄了 dispatch+GEMM1，没抄 stage2。

---

## 为什么「整核抄过来」仍然可以很快

`ep_s1_k` 就是 EP megakernel：~208 dispatch CU、grid=`num_cu * grid_mult`、contiguous A、48 个本地 expert。它快，是因为 **通信形态和 GEMM 布局是 EP 的**。现有 `chunk_fused_k` 慢，是另一套核（32 producer、token-id GEMM、chunk wait），不是「TP 物理上必须慢 8 倍」。

`ep_bcast_k` 在同一套融合核上只把 payload store 乘 8。那一列才是「为了让每卡都能算 expert0，XGMI 要多付多少」。它 **不能** 当成正确 TP：广播写进对端 `local_expert * cap + offset`，会和对方自己的 expert 槽打架。

真要做 TP，还得另做：dense AG 布局、384 expert 的 wait、N-shard GEMM、以及「只把需要的 token 发给 8 卡」而不是把 EP 槽位原样 ×8。
