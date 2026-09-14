# 增量 AG 融合：Step 5 之后的性能计划

日期：2026-09-14
分支：`cguo/megamoe`
前置：`docs/gemm1_incremental_ag_impl_plan.md`（Step 0–5）、`docs/gemm1_allgather_fusion_plan.md`（D1/D2/D7）

**一句话：e2e 比融合前慢是预期内的，先把账算清再改核。不要在 `tokens=64` 上拍板 overlap 或合核形态。**

---

## 现状（Step 5）

`tp_body` 已换成：

```text
本地 MXFP8 量化 + NCCL AG(ids, wts) + moe_sorting
→ fused incremental GEMM1（min-expert push + per-expert wait）
→ 已有 GEMM2 + NCCL RS
```

`bench.sh`（`--tokens 64 --mtpr 64 --iters 30 --route uniform --tp`，CUDA Graph 抽干流水线）：

| 路径 | mean / max |
|---|---|
| mori e2e | 0.3784 / 0.3806 ms |
| mega e2e | 0.3387 / 0.3392 ms |
| **tp_e2e** | **0.7678 / 0.7681 ms** |
| **tp_stage1** | **0.6154 / 0.6161 ms** |

`tp_vs_mori_rel_l2 = 5.88e-03`。路径按 Step 5 约定保留，即使更慢。

拆开看：`tp_stage1` 占 e2e 约 80%，sort + GEMM2 + RS 大约 0.15 ms。下一步只盯 stage1，不动 GEMM2、arena、sort-in-kernel。

`tokens=64`（全局 M=512）落在「通信不是瓶颈、协议开销主导」区（融合计划 D7）。384 个 expert flag 的 system atomic、合核 occupancy、以及 Graph 抽干都会把融合路径放大。这一档看不出 overlap 有没有赚钱。

---

## 原则

1. **先测量，再选 Step 6 的哪一项。** 慢了先做 window / 两发，不加更细握手。
2. **64 只作协议税，256+ 才看 overlap。** 与 impl 计划「小 M 看不出 overlap」一致。
3. **同一套 `do_tile` + token-id loader 比形态，不比无关 kernel。**
4. **仍不改** EP `mega_moe_stage1.py` / `dispatch.py`。不要把 `emit_tp_all_gather`（GEMM2 后 shard AG）当激活 AG。

---

## 下一步 1 — 把账算清（测，不改核）

同一套 `bench_mega_moe_v2.py`：

- 加回融合前基线（NCCL AG(x) + `fused_moe`），一次跑出 `tp_nccl` / `tp_fused`。
- `tokens ∈ {64, 256, 512, 2048}`。64 记录协议税，256+ 才解释 overlap。
- 把现在糊在一起的 `tp_stage1` 再拆：本地量化、ids/wts AG、fused 核。
- 受控 A/B（先验 D1）：同一套 `do_tile` + token-id loader，只比「push 然后 GEMM1」两发 vs 当前一发融合。

通过标准：每档有 `tp_nccl_e2e`、`tp_fused_e2e`、`tp_stage1` 分段，以及两发 vs 融合的核时间。没有这些数字不改核。

---

## 下一步 2 — 数字出来再选改法

| 如果看到 | 下一步 |
|---|---|
| 384 个 `received[e]` spin / atomic 占满 | **expert window**（若干 expert 一批减 flag）。impl 计划：慢了先做 window，不加更细握手 |
| 融合核比两发慢、occupancy / VGPR 掉 | **两发产品化**（`TPActivationGather` + `run_tp_gemm1`），不再坚持合核 |
| 大 M（512/2048）才通信受限、push 尾部拖 GEMM | 再考虑 **pull**（可删 system atomic；小 M 先验还可能更亏） |
| host 量化 / sort 派发各约 50–75 µs，且多层 back-to-back 才暴露 | 才考虑把 `moe_sorting` 收进核（方案 A，先验否过） |

热度降序可以跟 window 一起做，单独做收益不明。

---

## 明确先不做

- per-tile ready（TP 没有 tile 粒度 payload，融合计划 D5）
- GEMM loader 直读远端（流量约 18×，D3）
- GEMM2+RS 并 CCO arena（只影响 comm-fused 共存，降不了这 0.62 ms stage1）
- 在只看到 `tokens=64` 时宣布 overlap 失败或合核失败

---

## 正确性（速度分析阶段挂着）

`5.88e-03` 比「与 Mori 同量级」粗。多半是 MegaMoE SwiGLU epilogue vs 原来 fused_moe stage1，加上 v2 GEMM2 吃我们的 scale 布局。速度分析可以不管；一旦要宣布赢 NCCL，再对齐 epilogue / scale，避免把数值误差当成融合收益。

---

## 建议的下一手

只做「下一步 1」：基线开关 + 大 M + stage1 分段 + 两发 vs 融合。数字出来再决定是 window、拆核，还是 pull。
