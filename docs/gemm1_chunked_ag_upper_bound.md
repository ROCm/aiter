# AllGather 先发先算：dispatch 形态上限

日期：2026-09-17
前置：`docs/gemm1_incremental_ag_impl_plan.md`、`docs/gemm1_allgather_fusion_plan.md`

**一句话：融合方式定死为增量。先不排真正的 expert0，把 allgather 拆成和 EP dispatch 一样的多次有序发送，先发的先算，测 overlap 上限。真正把 expert0 token 排到第一波，后面再做。**

---

## 对照

EP MegaMoE 能 overlap，是因为：

```text
dispatch: expert0 打包连发 → payload_ready[0] → GEMM1 tile0
          expert1 打包连发 → payload_ready[1] → GEMM1 tile1
          ...
```

接收布局就是 GEMM1 的 A 布局，握手是 per-expert，不是整包 AG 再算。

现有 TP 增量（`emit_tp_incremental_payload`）已经是 min-expert 去重 + `received[e]`，但发送不像 dispatch：

- 8 个 producer 按 token 条带化，每个 token 打向全部 peer
- 每 token 对 topk 里每个 unique expert 做 system atomic（最多 384 个 flag）
- 真实 routing 下 expert0 的行并不在发送的最前面

这次改发送形态，不改 EP `mega_moe_stage1.py` / `dispatch.py`。

---

## 构造（先不管真实 expert 序）

本地行按 `0 .. m_local-1` 原样发（identity 顺序）。后面把真正的 expert0 token 排到这些前排行即可，kernel 不用改。

Allgather 拆成 C 次发送，C = ceil(m_local / chunk_rows)，默认 `chunk_rows = 32`（和 `sort_block_m` / dispatch tile 对齐）：

```text
for chunk in 0 .. C-1:          # 必须按序，chunk0 先发完再发 chunk1
    把本卡行 [chunk*S, (chunk+1)*S) push 到对端
        dest_row = rank * m_local + row
    fence_release
    peer.chunk_ready[chunk] += 1   # 每 rank 每个 chunk 只加一次；expected = npes
```

Producer 和 dispatch 一样 **按 destination 分片**（32 CTA，每卡 4 个 CTA 打一个 peer），不是按 token 条带打全网。

GEMM1 仍走现有 token-id loader（FLOPs 与现在相同）。Wait 按 **M-tile 下标** 映射到 chunk，不看真实 `expert_ids`：

```text
chunk(m_tile) = min(m_tile * C / n_m, C - 1)
```

于是 tile0 只等 chunk0 到齐就可以算。此时 A 里不一定是这个 tile 真正要的 token——**上限实验允许算错**。Kernel 结束时整包 AG 仍然写完，`rx` 应与 NCCL allgather 逐字节相同。

对照（正确性，不算 overlap）：所有 tile 都等最后一个 chunk。这时数值应与「bulk AG + GEMM1」一致。

---

## 上限在测什么

```text
two_launch     = gather_k + gemm1_k          # 不能 overlap
min-expert 融  = fused_k                     # 已有路径
chunk 先发先算 = chunk_fused_k               # 本次
```

理想流水线：

```text
chunk_fused ≈ T_chunk0 + max(AG - T_chunk0, GEMM1)
```

C 越大，第一波 GEMM 越早开工，handshake 也越多。默认 C = ceil(m_local/32)：

- tokens=64 → C=2，一半 AG 后一半 GEMM 可以叠
- tokens=256 → C=8

若 `chunk_fused_k` 仍明显慢于 `two_launch`，瓶颈就不是「expert 没排好」，而是合核 occupancy / 握手，后面排 expert0 也救不回来。

---

## 实测（2026-09-17，8×gfx950，7168×3072，e=384，k=6，uniform）

CUDA Graph 抽干，`--iters 10`，`chunk_rows=32`，32 个 dest-sharded producer。单位 ms（rank mean/max）。

| tokens | gather_k | gemm1_k | two_launch | min-expert fused_k | **chunk_fused_k** | max(gather, gemm) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.056 | 0.227 | 0.285 | 0.609 | **0.303** | 0.227 |
| 256 | 0.103 | 0.286 | 0.403 | 1.872 | **0.353** | 0.286 |

读法：

- tokens=64：AG 只有 ~56 µs，握手把 overlap 吃掉；chunk 融核略慢于两发（+18 µs）。和融合计划 D7 一致。
- tokens=256：chunk 融核 **0.353 < 0.403 两发**，叠掉大约一半 AG。离理论上限 `max(AG, GEMM)=0.286` 还差 ~67 µs（producer 占 CU + 合核 occupancy）。
- min-expert 那条（384 个 `received[e]`）在 256 上是 1.87 ms，握手本身就是瓶颈。chunk 路径只有 C=ceil(m_local/32) 个 flag。

正确性（`test_tp_chunk_fused.py`，小 shape）：wait-last 与 packed NCCL GEMM1 逐位相同；early-compute 的 `rx` 仍与 NCCL AG 逐字节相同，C=2 时 GEMM 约 0.3% 行对不上（先发先算、A 未齐，是预期）。

---

## 明确不做（本次）

- 按真实 topk 把 expert0 token 排到 chunk0（下一步）
- 改 EP dispatch / stage1
- 把 `moe_sorting` 收进核
- GEMM loader 直读远端
- 与 GEMM2+RS 共用 CCO arena
- pull 替代 push

---

## 落点

```text
aiter/ops/flydsl/kernels/mega_moe/tp_chunk_payload.py
aiter/ops/flydsl/kernels/mega_moe/tp_chunk_fused.py
op_tests/multigpu_tests/test_tp_chunk_fused.py
op_tests/multigpu_tests/bench_mega_moe_v2.py   # chunk_fused_k
```
