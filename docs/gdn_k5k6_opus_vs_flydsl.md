# GDN prefill 融合 K5+K6：opus WF vs FlyDSL VK (PR #4884)

MI308X · gfx942 · 80 CU · bf16 · K=V=128 · BT=64 · GQA 4（Hk=16 / Hv=64）· packed varlen
· 2026-08-26

对比的是两个**融合同一个边界**的实现：inter-chunk 状态扫描（K5）和输出投影（K6）在一次
dispatch 里完成。

| | opus WF | FlyDSL VK（PR #4884） |
| --- | --- | --- |
| 融合 kernel | `gdn_k2_kernel`（`gdn_k2_fused_traits`） | `chunk_gdn_fwd_h_o_flydsl_vk_{bv16,bv32,bv64w8}` |
| 入口 | `opus_gdn_wu_prefill_fwd(k2_mode=OPUS_GDN_K2_WU_FUSED)` | `chunk_gated_delta_rule_fwd_h_o_flydsl`，或 `chunk_gated_delta_rule_opt_vk(use_chunk_flydsl=True, fusion="always")` |
| 前端 K1..K4 | `gdn_k1_neumann_kernel`（一个 HIP kernel） | `gdn_prepare_kernel`（FlyDSL） |

---

## 1. 结论

**B·H（序列数 × 每卡 value head 数）是唯一移动结果的变量。** TP、seqlen、T 都只通过它起
作用：64 格里共享同一个 B·H 的格子彼此吻合到 2% 以内，即便 token 总量差 8 倍。

按 B·H 看融合 kernel 的 device time（把变体选择的 bug 修掉之后，见第 3 节）：

| B·H | 格数 | opus WF | FlyDSL | 谁快 |
| ---: | ---: | ---: | ---: | :--- |
| 8 | 1 | 1181 µs | 504 µs | **FlyDSL 2.35x** |
| 16 | 3 | 1185 µs | 638 µs | **FlyDSL 1.86x** |
| 32 | 6 | 902 µs | 558 µs | **FlyDSL 1.62x** |
| 64 | 10 | 617 µs | 747 µs | opus 1.21x |
| 128 | 12 | 1288 µs | 1603 µs | opus 1.24x |
| 256 | 12 | 1306 µs | 1434 µs | opus 1.10x |
| 512 | 10 | 2285 µs | 2676 µs | opus 1.17x |
| 1024 | 6 | 3220 µs | 4062 µs | opus 1.26x |
| 2048 | 3 | 4364 µs | 5538 µs | opus 1.27x |
| 4096 | 1 | 8683 µs | 10977 µs | opus 1.26x |

（µs 是该 B·H 下所有格子的中位数；不同格子的 seqlen 不同，所以列内绝对值不可横向比较，
比值是逐格算出来后取的 geomean。）

分界干净得没有例外：修正变体后 FlyDSL 赢的 10 格，正好就是 B·H ≤ 32 的那 10 格。
**B·H ≤ 32 → FlyDSL 快 1.6–2.4x；B·H ≥ 64 → opus WF 快 1.07–1.28x。**

这条分界和之前 WS/WF 的分界（B·H ≈ 40–48，见 `gdn_prefill_backend_perf.md`）位置几乎
重合，原因也是同一个：**融合 kernel 在低 B·H 时喂不饱设备**。区别在于两边低 B·H 的退化幅度
不同。取 seqlen=8192 固定链长，看融合 kernel 随 B·H 的走势：

| B·H | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| opus WF | 1181 | 1186 | 1205 | 1284 | 2562 | 5105 | 8944 |
| FlyDSL（修正变体后） | 504 | 644 | 743 | 1601 | 3209 | 5645 | 10558 |
| FlyDSL 的 CTA 数（fill） | 64 (0.8) | 64 (0.8) | 64 (0.8) | 128 (1.6) | 256 (3.2) | 512 (6.4) | 1024 (12.8) |

两边都是"到了某个 B·H 才开始按工作量收费"，只是拐点不同。opus WF 从 B·H=8 到 64 时间几乎
不动（1181 → 1284 µs），8 倍的工作量是免费塞进去的——反过来说，B·H=8 时它有 8 倍并行度没
用上。FlyDSL 把 V 轴切成 `⌈V/BV⌉` 份，靠调小 BV 把 CTA 数顶到 64（fill 0.8）一直撑到
B·H=32，所以在 B·H ≤ 32 区间它的时间只从 504 涨到 743 µs。B·H=64 起 CTA 数超过 CU 数，
两边都进入 throughput-bound，opus 的单 CTA 效率更高，稳定领先 1.1–1.3x。

生产上关心的那一点——TP=8、H=8、B=1、seqlen=8192（`varlen-64k-qwen-ptpc-ali` 的
`max_num_batched_tokens=8192` 档）——正是 B·H=8，**FlyDSL 融合 kernel 快 2.35x**
（504 vs 1181 µs），整块 pipeline wall 610 vs 1305 µs。

---

## 2. 两件影响解读的口径问题

**这台卡不是 PR 的目标卡。** PR #4884 把 `chunk.py` 里的 VK 路由门控在
`_device_cu_count() >= 304`（MI300X/MI325X），本机 80 CU，所以 `should_use_fused_gfx942()`
恒为 `False`——`fusion=AUTO` 在这里**永远不会融合**，默认路径跑的还是分离 K5 + Triton K6。
上面 FlyDSL 那一列是用 `fusion=ALWAYS` 强制打开的。换句话说，这份对比测的是"如果把 PR 的
融合 kernel 放到 MI308X 上会怎样"，而不是 PR 当前在 MI308X 上的行为（当前行为是：不启用）。

**`output_final_state` 两侧对齐了。** 两列都是 `output_final_state=True` 加真实
`initial_state`。PR 自己的 benchmark 在 review 里被 yiijin 指出 baseline 用 `True`、fused
候选用 `False`；这里不存在这个偏差，final state 的写回成本两边都算进去了。

数值上也核对过：FlyDSL 融合路径 vs 全 Triton 参考，`o` 的 max_abs 6.1e-5、`final_state`
的 max_abs 1.3e-4，和分离 FlyDSL K5 路径同一量级。

---

## 3. as-shipped 的变体选择规则在 80 CU 上是错的

第 1 节的 FlyDSL 数字来自"按 CU 缩放的变体"。如果照 PR 现在的规则跑，19/64 格会选到偏小的
BV，平均慢 1.96x、最坏 2.73x，曲线在 B·H=32/64 处塌陷成 opus 反超 1.7–2.1x：

| B·H | as-shipped 选 | CTA/CU | as-shipped | 应该选 | 修正后 | 提速 |
| ---: | :--- | ---: | ---: | :--- | ---: | ---: |
| 16 | `bv16` | 1.6× | 1005 µs | `bv32` | 638 µs | 1.56x |
| 32 | `bv16` | 3.2× | 1514 µs | `bv64w8` | 558 µs | **2.71x** |
| 64 | `bv32` | 3.2× | 1314 µs | `bv64w8` | 747 µs | 1.73x |

根因在 `aiter/ops/flydsl/kernels/chunk_gated_delta_h_gfx942.py`：

```python
_HN_BV32 = 32   # H*N above this prefers bv32 over bv16
_HN_BV64W8 = 80 # H*N above this prefers bv64w8

def _hn_variant(*, H, N, V):
    hn = H * max(1, N)
    if hn <= _HN_BV32:      tag = "bv16"
    elif hn <= _HN_BV64W8:  tag = "bv32"
    else:                   tag = "bv64w8"
```

这是**绝对阈值，不含 CU 项**。在 304 CU 上它们对应约一个 CTA wave（`H·N=32` 配 bv16 →
256 CTA，fill 0.84；`H·N=80` 配 bv32 → 320 CTA，fill 1.05），选得很准；搬到 80 CU 上，
同样的阈值要求最多 3.2 个 wave，于是 CTA 排队、时间线性翻倍。

单点探针（`probe_fused_variant.py`，seqlen=8192，强制四种变体）：

| B·H | auto | bv16 | bv32 | bv64 | bv64w8 | 最优 | auto 罚分 |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- | ---: |
| 8 | `bv16` | **506** | 640 | 999 | 735 | `bv16` | 1.00x |
| 16 | `bv16` | 1008 | **645** | 1001 | 737 | `bv32` | 1.56x |
| 32 | `bv16` | 2010 | 1275 | 1002 | **744** | `bv64w8` | 2.70x |
| 64 | `bv32` | 3654 | 2680 | 2152 | **1598** | `bv64w8` | 1.68x |
| 128 | `bv64w8` | 6815 | 4872 | 4325 | **3183** | `bv64w8` | 1.00x |
| 256 | `bv64w8` | 13651 | 9206 | 7629 | **5607** | `bv64w8` | 1.00x |
| 512 | `bv64w8` | 27407 | 18205 | 14193 | **10528** | `bv64w8` | 1.00x |

最优变体在每一行都是"网格 CTA 数最接近但不超过 CU 数"的那个，规则很简单：

```python
def cu_scaled_variant(bh, cus, V=128):
    for tag, bv in (("bv16", 16), ("bv32", 32), ("bv64w8", 64)):
        if -(-V // bv) * bh <= cus:
            return tag
    return "bv64w8"
```

`_fused_bv_for_shape` 里其实已经有等价的 fill-based 回退
（`_select_bv_for_grid(target_ctas=int(_GFX942_MIN_FILL * _device_cu_count()))`），只是被
硬编码的 `_hn_variant` 抢先返回了。**给 PR 的建议**：把 `_HN_BV32` / `_HN_BV64W8` 按
`_device_cu_count() / 304` 缩放，或在非 304 CU 的 gfx942 上直接跳过 tuned 表、落到已有的
fill 回退。按 80/304 缩放得到的阈值（8.4 / 21）和实测最优分界（8 / 16–20）吻合。

---

## 4. 整块 pipeline 的 wall

融合 kernel 之外，两条 pipeline 的前端也不同，而 FlyDSL 的前端更快：`gdn_prepare_kernel`
比 opus 的 `gdn_k1_neumann_kernel` 快 **1.30x**（geomean），这部分抵消了大 B·H 区间
K5+K6 的落后。

| 口径 | geomean(opus / FlyDSL) | FlyDSL 赢 |
| --- | ---: | ---: |
| K5+K6 融合 kernel，as-shipped 变体 | 0.766 | 4 / 64 |
| K5+K6 融合 kernel，CU 缩放变体 | 0.937 | 10 / 64 |
| 整块 pipeline wall，as-shipped | 0.918 | 25 / 64 |
| 前端 K1..K4 | 1.30 | 64 / 64 |

也就是说：即使带着变体选择的 bug，全 pipeline 口径下 FlyDSL 已经在 25/64 格上赢；
把变体修好之后 B·H ≤ 32 的区间会整体倒向 FlyDSL。

---

## 5. 复现

PR 分支和本地 opus varlen/GQA 改动合并后无冲突（改动面几乎不重叠：PR 只碰
FlyDSL/Triton 侧，本分支主要是 `csrc/opus_gdn` 和 `aiter/ops/gdn_prefill.py`）：

```bash
git fetch origin users/vpietila/kda-prefill-chunk-gated-delta
git checkout -b tmp/gdn-k5k6-compare
git merge FETCH_HEAD          # 无冲突
```

```bash
# 64 格网格：opus WF / FlyDSL auto / FlyDSL 指定变体
HIP_VISIBLE_DEVICES=7 python op_tests/flydsl_tests/sweep_k5k6_compare.py

# 变体探针：强制四种 BV，找每个 B·H 的最优
HIP_VISIBLE_DEVICES=7 python op_tests/flydsl_tests/probe_fused_variant.py

# 网页（就地刷新入库的那份）
python op_tests/flydsl_tests/render_k5k6_compare.py
```

产物：

| 文件 | 内容 |
| --- | --- |
| `op_tests/flydsl_tests/k5k6_compare.json` | 64 格原始数据（三列 × wall/k5k6/front + 变体） |
| `op_tests/flydsl_tests/fused_variant_probe.json` | 变体探针原始数据 |
| `docs/gdn-k5k6-opus-vs-flydsl.html` | 交互网页（悬停看每格明细，可切 as-shipped / CU 缩放） |
| `docs/images/gdn-k5k6-opus-vs-flydsl.png` | 网页截图 |

网格维度：TP ∈ {1,2,4,8}（H = 64/32/16/8，Hg = H/4）× seqlen ∈ {1K,2K,4K,8K}
× T ∈ {8K,16K,32K,64K}，B = T/seqlen，packed varlen。每格 wall 取 50 次中位、
per-kernel device time 取 profiler 20 次平均。
