# 将 FlyDSL 独立 GEMM 的 Pipeline 移植到 aiter MoE GEMM

## Context

aiter 的 MoE grouped GEMM kernel (`aiter/ops/flydsl/kernels/gemm_mxscale_gfx1250.py`) 和 FlyDSL 独立 GEMM kernel (`/app/FlyDSL/kernels/gemm_fp8fp4_gfx1250.py`) 共享相同的底层框架（`gemm_common_gfx1250` helpers、`make_tail_plan`、`@flyc.kernel`），但独立版的 pipeline 有多项性能优化尚未移植到 aiter 版。

**目标**: 将独立 GEMM 的 pipeline 优化搬到 aiter MoE GEMM kernel，同时保留 aiter 特有的 grouped/MoE 功能。

## 要移植的优化（独立版 → aiter 版）

### 1. `_pack_dg0` shuffle trick
- **源**: `gemm_fp8fp4_gfx1250.py:2331-2337`
- **效果**: 避免 LLVM 生成冗余 `s_mov_b64`
- **改法**: 在 aiter 版中，所有 `vector.from_elements(T.vec(4, T.i32), [pred, lds_addr, addr_lo, addr_hi])` 替换为 `_pack_dg0(pred, lds_addr, addr_lo, addr_hi)`
- **涉及位置**: prologue (L2123-2132, L2136-2170)、main loop WS callback (L2244-2252)、main loop NWS callback、tail plan callbacks

### 2. `readfirstlane` SGPR hoisting
- **源**: `gemm_fp8fp4_gfx1250.py:2452-2460`
- **效果**: `pred_const`, `active_addr_hi`, per-stage `lds_addr` 提升到 SGPR
- **改法**: 在 aiter WS-TDM prologue 前加 `readfirstlane` hoisting（仅 WS-TDM 路径）
- **涉及位置**: aiter L2121 之前

### 3. TDM issue BEFORE fence（WS-TDM 路径）
- **源**: `gemm_fp8fp4_gfx1250.py:2627-2642`
- **当前 aiter**: TDM 在 `mid_compute_callback` 中发射（fence 之后、compute 中间）
- **改法**: 主循环中将 TDM 发射移到 fence_signal/fence_wait 之前，addr advance 紧跟 TDM
- **涉及位置**: aiter WS-TDM main loop (L2218-2274)
- **注意**: 需要重构 `_mid_tdm_ws` → 提取到循环体顶部

### 4. `use_ws_tdm_split_signal_overlap`
- **源**: `gemm_fp8fp4_gfx1250.py:2583-2584, 2640-2650`
- **效果**: fence signal 延迟到 compute tile 的 `late_compute_callback`
- **改法**: 新增 `use_ws_tdm_split_signal_overlap` 编译参数；满足条件时 signal 从 tile 顶部延迟到 `late_compute_callback`
- **条件**: `wave_specialized_tdm and (fp8_quadrant or fp8_deep_pipeline) and num_buffers==4 and use_cluster`
- **涉及位置**: aiter WS-TDM main loop + tail

### 5. LDS prefetch interleaving (`_use_lds_pf`)
- **源**: `gemm_fp8fp4_gfx1250.py:2512-2579, 2586-2723`
- **效果**: 下一个 tile 的 ds_load 在当前 WMMA 间隙中发射
- **改法**: 新增 `_use_lds_pf` 路径（仅 `ROW_MAJOR_STREAMING` + WS-TDM + 非 BVS）
  - 新增 helper: `_pf_all_ks_to_flat`, `_flat_to_pf_all_ks`, `_issue_pf_all_ks`
  - 主循环 loop-carry 增加 pf_flat
  - `compute_tile_scheduled` 传 `pf_all_ks` + `next_lds_*`
  - tail plan 增加 pf chaining
- **涉及位置**: aiter WS-TDM main loop、tail plan

### 6. BVS ring（buffer-VGPR-scale）
- **源**: `gemm_fp8fp4_gfx1250.py:2500-2506, 2654-2667`
- **效果**: scale 绕过 TDM+LDS 直接 buffer_load→VGPR
- **改法**: 新增 `_bvs_active` 路径
  - prologue: prefetch `_bvs_D` K-tiles 的 scale
  - 主循环: ring buffer carry, 每 tile 消费 front + prefetch next
  - `compute_tile_scheduled` 传 `pf_a_scales`/`pf_b_scales`
- **前置**: 需要先移植 `_bvs_load_scales`, `_bvs_prefetch` helpers
- **涉及位置**: aiter WS-TDM prologue、main loop、tail

### 7. 地址简化（WS-TDM 路径去掉 addr_hi carry）
- **源**: `gemm_fp8fp4_gfx1250.py:2480, 2638` — 简单 `addr_lo + adv`，无 carry
- **当前 aiter WS-TDM**: 也是简单 add（L2133, L2256），已经一致
- **当前 aiter NWS**: 使用 `_advance_addr` 带 carry（L2114-2118）
- **改法**: NWS 路径保持不变（aiter 特有需求）；WS 路径已一致，无需改

## 必须保留的 aiter 特有逻辑

1. **`stage1_dual_b`**: 双 B 矩阵（gate+up）的 TDM load + 额外 compute_tile（NWS 路径）
2. **`_advance_addr` 64-bit carry**: NWS 路径的地址进位处理
3. **Grouped expert iteration**: persistent/contiguous/dense 三种模式的 grid dispatch 和 batch_idx 计算
4. **`stage1_act` fused epilogue**: SiLU/SwiGLU 激活融合
5. **`epilogue_bias`**: per-expert bias
6. **`grouped_masked_m` / `m_tile_prefix` / `m_tile_map`**: per-expert variable M

## 实施进度

> **状态快照（2026-06-17 复核）**：Phase 1 / 3 / 5 / 6 均**已实现并提交**（HEAD `eeb555f2b merge continue`，工作区干净）。Phase 2 fp8-only 子项确认不适用已跳过。**Phase 4 (BVS ring) 是唯一未实现项**。

- **Phase 1 ✅ 已完成 + 已验证**（2026-06-17）
- **Phase 3 ✅ 已完成（已在代码中）**：`_use_lds_pf`（kernel L2383）、helpers `_pf_all_ks_to_flat`/`_flat_to_pf_all_ks`（L2406/2415）、`_issue_pf_all_ks`（L1081）、主循环 pf loop-carry（L2434-2546）、`compute_tile` 三分支（`_has_pf and _has_next` 交错 L1148 / `_has_pf` tail L1188 / 旧路径回退）全部就位。开关 `AITER_GROUPED_LDS_PF=1`（默认 on，仅 WS-TDM 路径生效）。
- **Phase 5 ✅ 已完成（已在代码中）**：tail `_tail_plan_ext` 预展开 next-cs + `_use_tail_pf` 链式 prefetch（L2845-2999），`loop_iters==0` 的 seed。
- **Phase 6 ✅ 已完成**（`_pack_dg0` 替换 NWS 路径，与 Phase 1 一并做）
- **WS-TDM 启用开关 ✅ 已加**：`AITER_GROUPED_WS_TDM=1` 让 stage2 走 WS-TDM（`grouped_moe_gfx1250.py:969-994`）。stage1 因 act 融合自动禁用；要求 `m_warp*n_warp==4` 且 `split_k2==1`。
- **WS-TDM 已通过 ISA 验证真正启用**（2026-06-17）：对比 `kernel_mxscale_gemm2` 的 final ISA，WS-on vs WS-off：`tensor_load` 16→4（每 wave 专精一个 tensor）、`readfirstlane` 37→5（Phase 1 SGPR hoisting 生效）、`s_mov_b64` 5→2（`_pack_dg0` 生效）、ISA 1139→950 行。数值 rel_l2=2.74e-3 与 NWS 一致。
- **WS+LDS-PF 数值正确性已复验**（2026-06-17，GPU1）：a4w4 swiglu, model_dim=7168 inter=512 E=32 tok=64 topk=4。baseline(WS off) rel_l2=2.6906e-3 == WS+PF on rel_l2=2.6906e-3（完全一致）。
- **性能 A/B（PF on vs off）⏳ 未完成**：a8w4 E=128 tok=1 dim=3072/3072 bench 在 GPU1 上 JIT 卡住（22min 无输出后手动 kill）。待换更小 E 或用已验证的 a4w4 shape 重测。

### a-scale（激活 scale）布局事实（a8w4 fast path，`grouped_moe_gfx1250.py`）

- **quant 始终输出 row-major，`shuffle=False`**（不在 quant 阶段 shuffle）。a1: `_quantize_mxfp8_payload`→`dynamic_mxfp8_quant`（L741）；a2 同（L939）。
- **布局变换是单独一步**，变成 WMMA 交错布局。fast path = **fused gather+preshuffle**：stage1 `flydsl_moe_scatter_preshuffle_scale`（L778，route-gather + preshuffle 一个 kernel）；stage2 `flydsl_moe_preshuffle_scale`（L951，a2 已 grouped 无需 gather）。naive path = `_grouped_a8w4_preshuffle_e8m0_scale`（L254）：`permute(0,1,3,4,5,2,6)` 把 wmma_rep 轴和 lane16/k 交错。kernel 经 `_precompute_scale_lane_bases` + `interleaved_scale_cols_a` 读取。
- 所以「fused shuffle」指 **gather+preshuffle 融合**，不是 quant+shuffle 融合。布局**确实改变**（row-major → WMMA 交错）。
- **对 Phase 4 (BVS) 的影响**：BVS 要 a-scale 绕过 LDS 直接 `buffer_load→VGPR`，其 lane→address 映射**必须匹配此 preshuffle 布局**。FlyDSL `_bvs_load_scales` 用自己的 segmented 布局，与 aiter 这个 permute 布局不一定对得上——即计划中 Phase 4「前置：aiter 版可能不满足 segmented LDS layout」的具体含义。

### Phase 2 关键发现（重要架构错配）

调查后发现计划与 aiter 实际架构有错配，**Phase 2 的多数子项对 aiter 不适用**：

1. **「TDM before fence」只在 WS 路径**：FlyDSL 的 NWS 路径同样把 TDM 放在 `_mid_tdm_nws` callback 里（fence 后），与 aiter 现状**完全一致**，无需改。aiter 的 WS 路径目前仍用 `_mid_tdm_ws` callback（Phase 1 已统一走 `_issue_active_tdm`），可选择性提前。
2. **`use_ws_tdm_split_signal_overlap` + `late_compute_callback` + `a0_prefetch` 依赖 fp8 schedule**：FlyDSL 这些只在 `compute_tile_fp8_quadrant` / `compute_tile_fp8_deep_pipeline` 里实现。aiter **只有 `ROW_MAJOR_STREAMING` 和 `FP4_COL_BAND` 两种 schedule，没有 fp8 schedule**，移植后是 dead code。
3. **真正有移植价值的是 Phase 3**：`_use_lds_pf`（LDS prefetch interleaving）针对 `ROW_MAJOR_STREAMING`，aiter 有这个 schedule，但需要给 aiter 的 `compute_tile` 加 `pf_all_ks` / `next_lds_*` 支持。

**结论**：Phase 2 跳过 fp8-only 子项（4、5），核心「TDM 提前」对 aiter WS 路径可做但收益有限（NWS 已一致）。重点转向 **Phase 3**（LDS prefetch），这是 aiter 上唯一有明确性能潜力的移植项。

## 实施顺序

### Phase 1: 基础工具函数（低风险）— ✅ 已完成
1. ✅ 添加 `_pack_dg0` helper 到 aiter 的 `gemm_mxscale_gfx1250.py`
2. ✅ 将所有 `vector.from_elements(T.vec(4, T.i32), [...])` 构建 dg0 的地方替换为 `_pack_dg0`
3. ✅ 添加 `readfirstlane` SGPR hoisting（WS-TDM prologue 前）
4. ✅ 添加 `_issue_active_tdm` helper（封装 WS-TDM load 发射）

### Phase 2: WS-TDM 主循环重构 — ⚠️ 大部分不适用 aiter（见上方关键发现）
1. ⏭️ TDM 提前到循环顶部：NWS 已与 FlyDSL 一致；WS 路径可选做，收益有限
2. ⏭️ addr advance 紧跟 TDM：同上
3. ⏭️ fence_signal/wait 时序：aiter 现状已合理
4. ❌ `use_ws_tdm_split_signal_overlap`：依赖 fp8 schedule，aiter 无，跳过
5. ❌ `a0_prefetch` / `maybe_prefetch_fp8_deep_a0`：依赖 fp8 deep pipeline，aiter 无，跳过

### Phase 3: LDS prefetch 路径（WS-TDM only）— ✅ 已完成（已在代码中）
1. ✅ 添加 `_use_lds_pf` 条件判断
2. ✅ 添加 `_pf_all_ks_to_flat`, `_flat_to_pf_all_ks`, `_issue_pf_all_ks` helpers
3. ✅ 主循环增加 pf_flat loop-carry
4. ✅ prologue drain + barrier + 初始 pf issue
5. ✅ tail plan 增加 pf chaining（`_tail_plan_ext`）

### Phase 4: BVS ring（可选，最复杂）— ❌ 未实现（唯一剩余项）
1. 添加 `_bvs_load_scales`, `_bvs_prefetch` helpers
2. prologue BVS prefetch
3. 主循环 BVS ring carry
4. tail BVS handling
- **前置阻塞**：FlyDSL `_bvs_load_scales` 用 segmented 布局，aiter a-scale 用 `permute(0,1,3,4,5,2,6)` WMMA 布局，BVS lane→addr 映射须先确认能匹配（见上方「a-scale 布局事实」）。
- **决策建议**：先拿到 WS+LDS-PF 的性能数据；若流水线已喂饱，BVS 的寄存器压力（spill 风险）可能不值得。

### Phase 5: Tail plan 更新 — ✅ 已完成（已在代码中）
1. ✅ tail plan 的 TDM 发射时序（`_tail_mid_ws` / `_tail_mid_nws`）
2. ✅ WS-TDM tail TDM 发射（匹配主循环）
3. ✅ LDS pf chaining: `_tail_plan_ext` + `_use_tail_pf`

### Phase 6: NWS 路径（可选）
- NWS 路径也可以用 `_pack_dg0` 替换 `vector.from_elements`
- 但 NWS 的 `_mid_tdm_nws` callback 结构保持不变（有 `stage1_dual_b` 依赖）

## 修改文件清单

| 文件 | 改动 |
|------|------|
| `aiter/ops/flydsl/kernels/gemm_mxscale_gfx1250.py` | 主要修改：pipeline prologue/main-loop/tail/epilogue |
| `aiter/ops/flydsl/kernels/gemm_common_gfx1250.py` | 可能需要检查 helper 兼容性（应该无需改） |

## 验证

1. **编译测试**: 确保 kernel 能正常编译（`compile_a8w4_gemm` / `compile_mxfp4_gemm`）
2. **单元测试**: 运行现有 MoE GEMM 单元测试
3. **数值正确性**: 对比改动前后的 `grouped_a2`（stage1 输出）和 `grouped_out`（stage2 输出）
4. **性能**: 用 `rocprof` 对比改动前后的 kernel 执行时间
5. **环境变量覆盖**: 测试 `AITER_GROUPED_DEEPGEMM_CONTIGUOUS=1`、`AITER_GROUPED_GEMM_NAIVE=1` 等路径

## 风险

- **Phase 2 是核心**：TDM 发射时序改变可能引入 data race（需要确保 fence 正确）
- **LDS prefetch 增加 loop-carry state**：VGPR 压力增大，可能导致 register spill
- **BVS 前置条件**：需要 reference segmented LDS layout，aiter 版可能不满足
