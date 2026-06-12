# FlyDSL stage2 gemm2 持久化 N-loop kernel —— 实现设计文档（方案 B）

> 目标：新增一个 `_persist` 变体的 flydsl stage2 down-projection kernel，对齐 asm `moe_2stage_down` 的"少量长 WG + WG 内 loop 整个 N + 跨 N-tile 预取 + A(X) 复用"结构，隐藏 W2 读延迟，提升达成带宽。env 默认关、独立验证、不动现有默认路径。

---

## 1. 动机（rocprof-compute 已确诊）

reduce t32x256 gemm2 = 4684us。SOL：ALU 活跃 15.9%、指令等待 35.3%、达成 HBM 带宽 ~10% 峰值、占用 97%、LDS 冲突可忽略、写 100% 合并+nt。

**真瓶颈**：W2 读延迟没被藏住（即便 79% 命中 L2，浅流水 + 短 WG 让 W2 load 延迟暴露）。

**asm 对照**：grid=1D(M-blocks)，每 WG=256 线程，**WG 内 loop 整个 N=4096（32 个 N-tile）**，VGPR=208 深流水，达成带宽 ~14%（1.5×）。asm 的 WG 长时间运行，能跨 N-tile 把 k+1 的 W2 load 与 k 的 compute/write 重叠 → 藏住读延迟。

**flydsl 现状**：grid=2D(gx=16 N-tiles, gy=M-blocks)，每 WG 只做 1 个 N-tile、跑完即退 → 无跨 N-tile 流水机会，W2 读延迟暴露。

---

## 2. 现状结构（compile_moe_gemm2，相关行）

- `by = block_id("x")`（N-tile），`bx = block_id("y")`（M-block）。grid=`(gx=model_dim/tile_n, gy=size_expert_ids, k_batch)`，block=`total_threads`(=num_waves*64)。
- 单 WG body（`_moe_gemm2_then_body`）：
  1. expert_id / expert_off（per M-block）
  2. X(A2) gmem→reg→LDS，K 循环 **ping-pong** 加载 X + B，MFMA 累加（`compute_tile`）
  3. epilogue：`c_shuffle_epilog`（CShuffle LDS 重排 + dequant + store/atomic）
- `by_n = by*tile_n`（2835）往下全部依赖 N-tile：`col_g_list/n_blk_list/n_intra_list`、`load_b_tile`、epilogue 闭包（`write_row_to_lds`/`store_pair` 捕获 by 派生值）。
- sorted_token_ids 已预载 `lds_tid`（per M-block，可复用）。
- LDS：`lds_x`(ping-pong, `2*tile_m*lds_stride*elem`) 与 `lds_out`(`2*tile_m*tile_n`, CShuffle) **共用同一段 LDS**（X 先消费完才用 lds_out，可 alias）。`lds_tid` 在其后。

---

## 3. 目标结构（_persist）

```
# ===== per-M-block setup（loop 外，做一次）=====
expert_id, expert_off
preload sorted_token_ids -> lds_tid
load FULL X (所有 K=192) -> lds_x_persist   # 单缓冲、非 ping-pong、跨 N-tile 复用
barrier

# ===== N-tile 持久化循环（WG 内 loop 整个 N）=====
b_next = load_b_tile(by=0)                    # 预取第 0 个 N-tile 的 W2
for by in range(gx):                          # gx = model_dim/tile_n 个 N-tile
    b_cur = b_next
    if by+1 < gx:
        b_next = load_b_tile(by+1)            # 跨 N-tile 预取下一个 W2（关键！）
    by_n = by * tile_n
    重算 col_g_list / n_blk_list / sw_vals（依赖 by_n）
    acc = zeros
    for k_tile in range(num_k_tiles):         # K 循环：只载 B 已在手；A 从 lds_x_persist 读
        acc += mfma(A_from_lds, b_cur[k_tile])
    epilogue(by_n): CShuffle + dequant + store
```

要点：
- **X 一次性载入、复用**：去掉 K 循环里的 X ping-pong，X 在 N-loop 外全量入 LDS（K=192 全部）。K 循环只读 LDS 里的 A、配合当手的 B。
- **跨 N-tile 预取 B**：`b_next = load_b_tile(by+1)` 在用 `b_cur` 计算之前发射 → 用 N-tile 间的独立性让 W2 load 与 compute/write 重叠（这是隐藏读延迟的核心，且 N-tile 间无累加依赖、比 K-loop 内预取更易被 JIT 生成）。
- grid 改 `(1, gy, k_batch)`（N 不再进 grid），block 不变。

---

## 4. LDS 布局（关键约束：X 与 lds_out 不能再 alias）

持久化下，X 要在整个 N-loop 存活，而每个 N-tile 的 epilogue 要用 lds_out（CShuffle）→ **二者同时存活，不能共用 LDS**，必须分开分配：

| 段 | 大小（t32x256, fp8 例） | 说明 |
|---|---|---|
| `lds_x_persist` | `num_k_tiles * tile_m * lds_stride * elem` = 3*32*64 = **6144 B** | X 全量（所有 K），单缓冲 |
| `lds_out` | `2 * tile_m * tile_n` = **16384 B** | CShuffle epilogue（每 N-tile 复用） |
| `lds_tid` | `tile_m*4` = **128 B** | sorted_token_ids |
| **合计** | **≈ 22.7 KB** | < 64KB/CU ✓ |

对比现状 16.9KB：LDS 升到 ~22.7KB。gfx942 64KB/CU → LDS-occupancy 从 3→2 WG/CU，但本 kernel 非占用受限、且 persist 本就 WG 少，可接受。

实现：新增 `lds_x_persist` 独立分配（不与 lds_out alias），其余 lds_out/lds_tid 保持。X 读取仍用现有 `lds_load_packs_k64` + `swizzle_xor16`（布局不变，只是改成"载一次、读 N 次"）。

---

## 5. 改造步骤（实现顺序，每步可独立验证）

**Step A — 加 env 开关 + dispatch 接线**
- `compile_moe_gemm2(..., persist: bool=False)`；kernel 名加 `_persist` tag（`moe_kernels._parse_flydsl_kernel_name` 解析 `persist` token → `params["persist"]=True`，已有 stage2 `persist` 解析，复用）。
- `compile_flydsl_moe_stage2` 透传 persist 到 `compile_moe_gemm2`。
- grid：persist 时 `gx_launch=1`，kernel 内 loop gx。

**Step B — X 全量载入 LDS（去 ping-pong）**
- 新增 `load_full_x_to_lds()`：把现有 `load_x_tile`+`store_x_tile_to_lds` 对所有 K-tile 跑一遍，写入 `lds_x_persist`（单缓冲）。
- 在 N-loop 外调用一次 + barrier。

**Step C — N-loop 骨架（先不预取，验正确性）**
- 把 by_n/col_g/n_blk/sw_vals/K循环/epilogue 包进 `scf.ForOp(0, gx, 1)`，by=induction var。
- K 循环改为：B 当手载入，A 从 `lds_x_persist` 读（`lds_load_packs_k64` 指向 persist 缓冲、对应 K-tile 偏移）。
- epilogue 闭包改为每 N-tile 重建（by_n 为循环变量）。
- **先验正确性**（cos vs 参考），此时性能可能持平/略差（无预取）。

**Step D — 跨 N-tile B 预取**
- `b_next = load_b_tile(by+1)` 提前发射；`_barrier(vmcnt=_num_b_loads)` 部分等待。
- 用 scf.ForOp 的 iter_args 传递 `b_next`（或在循环体内预取下一轮）。
- 这步是性能关键，验 gemm2 us + rocprof 达成带宽。

**Step E — 调优**
- waves_per_eu / b_nt / 预取深度（预取 2 个 N-tile？）；rocprof-compute 复测 ALU 活跃、inst-wait、达成带宽。

---

## 6. 正确性要点

- **sentinel/padding**：N-loop 不改 M 方向的 padding 语义；sorted_token_ids 掩码逻辑（epilogue 的 `precompute_row`/`write_row_to_lds`）保持，每 N-tile 都跑（行有效性与 by 无关）。
- **blk_valid 守卫**：仍在 N-loop 外（依赖 bx，与 by 无关）——整块无效则跳过全部 N-tile。
- **reduce 模式 target 写位置**：`out_base + ts_idx*model_dim + col_g0`，col_g0 含 by_n → N-loop 内每个 by 写不同 N 段，天然正确、无重叠。
- **W2 资源索引**：`n_blk/n_intra` 依赖 col_g(by_n)，每 N-tile 重算。
- **X swizzle 一致性**：persist 载入与读取用同一 `swizzle_xor16`，与现状一致。

---

## 7. dispatch / 配置

- kernel 名：`flydsl_moe2_afp8_wfp8_bf16_t32x256x64_reduce_persist_bnt0`（加 `_persist`）。
- env：`FLYDSL_MOE_STAGE2_PERSIST`（默认 0）作为额外保险；或纯靠 kernel 名 tag。
- 落地条件：correctness pass 且 gemm2 us < 4684（reduce 基线）才切默认；否则保留为可选、默认关。

---

## 8. 风险与回退

| 风险 | 缓解 |
|---|---|
| 热 kernel 重写引入正确性 bug | 分步（A-E），每步 `check_result`（cos≥0.999） |
| 跨 N-tile 预取 JIT 生成不出来（同 deep-B 失败） | Step C 先验正确性，Step D 单独量收益；不行回退 |
| LDS 升到 22.7KB 降占用 | 本 kernel 非占用受限；persist WG 少 |
| 收益不及预期 | 全程 env 默认关，不动现有 reduce t32x256 默认 |

**回退**：env 默认关 + 独立 kernel 名，默认路径零影响。

---

## 9. 预期

- 乐观：跨 N-tile 预取藏住 W2 读延迟 → 达成带宽 10%→~14%（asm 水平）→ gemm2 4684→~3500-4000us（-15~25%）。
- 保守：JIT 仍生成不出深预取（同 deep-B）→ 持平或略差 → 回退。
- 这是唯一对齐 asm 架构、且打在实测真瓶颈（W2 读延迟未隐藏）上的杠杆；值得一试，但不保证赢。

---

## 10. 实现进度

### ✅ Step A（plumbing，已完成，默认路径零影响）
- `compile_moe_gemm2(persist=False)` 参数 + env `FLYDSL_MOE_STAGE2_PERSIST`（默认 0）。
- 约束守卫：persist 仅限 sync(非async)/非int4/非split-K。
- module 名加 `_persist` tag（独立编译缓存）。
- **LDS 布局**：persist 时 `lds_x`（全 K-tile 单缓冲）与 `lds_out` 分开分配、不再 alias（`_lds_out_byte_off=lds_x_bytes`）。实测 LDS 22656B（=设计的 22.7KB）。
- grid：persist 时 X 维 = 1（`gx = 1`），N-tile 在 WG 内 loop。
- 默认（persist OFF）实测不变：gemm2 4683.8、cos 1.0 ✓。

### ✅ Step C（N-loop 骨架，已完成、正确、有小幅收益）
- 把 body `_moe_gemm2_then_body(by)` 参数化（by 仅在 `by_n=by*tile_n` 用到一处）。
- 调用点：persist 时 `for _by_idx in range_constexpr(model_dim//tile_n): _moe_gemm2_then_body(fx.Index(_by_idx))`（编译期展开 16 个 N-tile）。
- **实测（reduce t32x256，cos 1.0 正确）**：gemm2 **4628 vs 4685 基线（-1.2%）**，稳定可复现。
- **rocprof 验证方向正确**：inst-wait 35.3%→**30.7%**（少等待）、ALU busy 15.9%→**22.7%**（计算更密）。arch_vgpr 52→68（16× 展开）、SQ_WAVES 大降（grid 1D，WG 少很多）。
- 注：此版 X 仍每 N-tile 重载（无复用），且每 N-tile 有 barrier 串行化 → 收益受限。

### ✅ Step B（X 复用，已完成 —— 关键增益落地）
- body 加 `is_first_ntile` flag。persist 分支（在 prologue 处 `if _persist:`）：
  - 首个 N-tile 把全 K-tile 的 X 一次性载入**静态 LDS 槽**（0, tile_elems, 2*tile_elems），`_barrier`。
  - 简单**静态-读 K 循环**：每 K-tile `load_b_tile` + 从静态槽读 A + `compute_tile` 累加；无 ping-pong、无 X store、无 per-N-tile X barrier。
  - 后续 N-tile（is_first_ntile=False）跳过 X 载入，直接读静态槽。
- 主循环用 `pair_iters = 0 if _persist` 跳过；tail 用 `if _persist: pass` 跳过（小改、无大段 reindent）。
- **实测（reduce t32x256，cos 1.0 正确，稳定）**：gemm2 **4102 vs 基线 4685（-12.4%）**，e2e **9299 vs 9878（-5.9%）**。
- **rocprof**：ALU busy 15.9%→**26.0%**（计算密度大增）、VGPR 52→92（静态 X + 3 K-tile 在飞）。收益来自去掉 16× 冗余 X 载入 + 去掉 per-N-tile X-load barrier 的串行化。
- **这是多轮结构性尝试后第一个真正的结构性胜利。**

### ⏳ Step D（跨 N-tile W2 预取，待做）
- 现状 inst-wait 仍 38.8%（W2 读延迟仍部分暴露）。预取下一个 N-tile 的 W2、跨 N-tile 软件流水可进一步隐藏。
- 阻塞：N-loop 在调用点展开，每个 body 自带 epilogue barrier，跨 N-tile 的 B 预取被 epilogue barrier 隔断。需把 N-loop 移入 body 并跨 epilogue 软件流水 —— 较深改造。

### ✅ Step E（dispatch 接线 + 落地，已完成）
- 接线链路：kernel 名 `_persist` token → parser `params["persist"]` → fused_moe `persist=parsed.get("persist")` → `flydsl_moe_stage2(persist=)` → `compile_flydsl_moe_stage2(persist=)` → `compile_moe_gemm2(persist=)`。
- 已把 `hy3_fp8_pertensor_tuned_fmoe.csv` token=32768 行的 kernelName2 改为 `flydsl_moe2_afp8_wfp8_bf16_t32x256x64_reduce_persist_bnt0`，**默认生效、无需 env**。实测 gemm2 4101、e2e 9295、cos 1.0。
- 非-`_persist` kernel 名零影响（实测 reduce 非 persist 仍 4686）。

### 最终落地结果
| | gemm2 us | e2e us |
|---|---|---|
| 原始 atomic | ~5892 | — |
| reduce t32x256 | 4685 | 9878 |
| **reduce t32x256 + persist（落地）** | **4101（-12.4% vs reduce, -30% vs 原始）** | **9295（-5.9%）** |

### 已穷尽的后续微调
- persist + waves_per_eu (w2/w4/w5)：无差别（均 ~4101），占用非瓶颈。
- gating lds_tid 预载（去 16× 冗余）：**反而 -2%**（4101→4183），调度器对去冗余敏感，已回退。

### 当前瓶颈（rocprof DRAM）
persist 写占比 90%→**86.9%**（X 载一次 + W2 在 WG 内复用 → read-DRAM 降），达成写带宽 514→**585 GB/s**（asm 760）。**现在是写受限**（2.4GB reduce partial 写，reduce 模式固有）。

### ✅ Step D（跨 N-tile W2 预取，已完成 —— 又一档增益）
- 实现：把 B-loader 参数化（`load_b_pack`/`load_b_tile` 加可选 `n_blk_l/n_intra_l`）+ 加 `_compute_nidx_for(by_val)` 算任意 N-tile 的 W2 索引。body 加 `b_preloaded`/`next_by` 参数并 `return` 预取的下个 N-tile B。
- 调用点软件流水（`_persist_pf`）：`b_pf=None; for nt: b_pf = body(nt, is_first, b_preloaded=b_pf, next_by=nt+1)`。body 在 K 循环后、**epilogue 写之前**发射下个 N-tile 的 W2 load → 读与当前 epilogue 写重叠。
- env `FLYDSL_MOE_STAGE2_PERSIST_PF`（persist 下**默认开**，=0 可关）。
- **实测（cos 1.0 正确、稳定）**：gemm2 **3934 vs persist-no-pf 4100（-4%）**，e2e **9126 vs 9291**。VGPR 仍 92（编译器很好地复用了寄存器，无暴涨）。
- 推翻了"write-bound 下预取无效"的负面先验——因为这次是**读与写重叠**（不是单纯堆 in-flight 读），填补了写突发期间的空闲读带宽。

### 最终落地结果（全部 cos 1.0、默认生效、默认路径零影响）
| 阶段 | gemm2 us | e2e us | 相对 reduce 基线 |
|---|---|---|---|
| 原始 atomic | ~5892 | — | — |
| reduce t32x256 | 4685 | 9878 | — |
| + persist（X 复用） | 4100 | 9291 | -12.5% |
| **+ Step D 跨 N 预取（落地）** | **3934** | **9126** | **-16.0%（vs 原始 atomic -33%）** |

vs asm 核 3162us：差距从最初的 1.86× 收窄到 **1.24×**。

### ✅ Step F（persist + atomic：端到端最优，已落地）
关键再发现：之前比较是 gemm2-only。按**端到端**看，reduce 模式虽 gemm2 低（3936），但要额外的 `topk_sum` 规约 kernel（~673us）；atomic 模式直接原子累加到 `[tokens, model_dim]`、**无需 topk_sum**。persist（X 复用 + 跨 N 预取）让 atomic 的 gemm2 从 5416→**4401**，于是：

| 模式（均 persist+pf, t32x256） | gemm2 | topk_sum | e2e | cos |
|---|---|---|---|---|
| reduce | 3936 | +673 | 9128 | 1.0 |
| **atomic（落地）** | **4401** | **0** | **8909（-2.4%）** | 0.99999 |

**atomic 端到端更优**（省掉独立规约 kernel），精度 cos 0.99999（与最初 atomic 基线一致，max_delta 0.0117 可接受）。tuned 配置改为 `t32x256_atomic_persist`。**e2e 8909 vs asm 1-stage 8602 = 1.036×**（已非常接近）。

### rocprof-compute 分析（确认非 VALU 受限）
persist+pf reduce SOL：VALU Util **52.9%**（最忙 pipe）、MFMA 26.1%、IPC **0.86**（低）、占用 92%、LDS 冲突 0.27。两次 VALU 削减实验印证非 VALU 受限：
- fast-valid-block（削 583 条 cmp/cndmask）：3966（更差）。
- v_pk dequant（640→320 muls）：3957（更差）。
> VALU 虽是最忙 pipe，但 IPC 低 = 停顿主导、VALU 不在关键路径。reduce gemm2 的地板是写带宽（610 GB/s vs asm 760）。

### 实验记录（已穷尽的微调）
- waves_per_eu / gating lds_tid / K-loop 全 B 前置 / 预取移到 K-loop 前 / 更大 tile / v_pk dequant / fast-valid：均无效或反伤，已回退。

### 落地状态
`hy3_fp8_pertensor_tuned_fmoe.csv` token=32768 行用 `..._reduce_persist_bnt0`，persist+pf 默认生效、无需 env。非 `_persist` kernel 名零影响。lint 通过、正确性 cos 1.0。env 旋钮：`FLYDSL_MOE_STAGE2_PERSIST`(默认按 kernel 名)、`FLYDSL_MOE_STAGE2_PERSIST_PF`(persist 下默认开)。
