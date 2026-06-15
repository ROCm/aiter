# Plan: a8w4 weight-scale → K-pairing（完全替换现有 a8w4 读法）

## 目标 / Goal
**a8w4 的 N4K8 weight-scale 读法全部替换为 K-pairing**：一条 `ds_load_b32` 拿到一个
N-tile 两个 WMMA-K=128 步的 scale（lane 0-15 = K-step0，lane 16-31 = K-step1），
`matrix_a_scale` ROW0/ROW1（op_sel）选 K-step，两个 K-step 累加到**同一个累加器**。
**删除** 现有 a8w4 的 per-tile（op_sel off）/ per-pair（op_sel on，N-tile 配对）两条读法
及其开关 `WEIGHT_SCALE_OP_SEL`。

Replace a8w4's N4K8 weight-scale read entirely with **K-pairing** (op_sel selects the
K-step; both K-steps accumulate into the same acc). Remove the per-tile / per-pair
reads and the `WEIGHT_SCALE_OP_SEL` toggle.

## 范围 / Scope
- **仅改 a8w4**（`is_a8w4`，`compute_tile` / ROW_MAJOR_STREAMING / `16x16x128_f8f6f4`）。
- **a4w4/fp4 不动**（`compute_tile_fp4_bank_friendly` / `32x16x128_f4`，32 行占满 wave，
  `scaleAType=0`，无 op_sel）。
- **fp8 不动**（`interleaved` / `load_scale_b128`）。

## 不需要改 / No change
- **Producer** `_grouped_b_scale_preshuffle_e8m0`：列 `remain_b*512 + p*128 + q*64 +
  lane16*4 + r` 已使每个 N-tile 的 `(q[2],lane16[16],r[4])`=128B 连续，正是 K-pairing 布局。
- **`make_desc_bs`(n4k8)**、**`adv_bs_i32`**（每 k-tile 仍 512B）、**shape 检查**
  `_preshuffled_b_scale_shape`。

## 核心机制 / Mechanism
a8w4-K-pairing 的 b-scale **与 ks 无关**（两个 K-step 在同一 dword 的两个 lane-half）。
→ 在 `compute_tile` 顶部**只加载一次**，per-ks 流水线（prologue/steady/tail、
partial-drain、scheduler）结构不变，`_load_b_and_scales` 通过闭包复用这组 SSA 值
（不再每 ks 发 b-scale ds_load）。

## 改动清单 / Changes（除注明外都在 `gemm_mxscale_gfx1250.py`）
新开关常量：`b_kpair = b_n4k8 and is_a8w4`。

1. **`_precompute_b_scale_n4k8_base`** (~L1000)
   - `b_kpair`：`base = super_local*ROW_BYTES + lane16*4 + lane_kgrp*64`
     （lane_kgrp → q/K-step 的 lane-half，stride 64）。
   - fp4：保持 `+ lane_kgrp*128`。

2. **K-pairing 读取**（a8w4 在 `_load_b_scale_n4k8` 的新分支）
   - 每 N-tile 一条 `ds_load_b32`：`off = scale_base + wn*128`（p stride，**无 q_off**）。
   - 返回 `wmma_n_rep` 个 dword，按 `wn` 索引，**与 ks 无关**。

3. **`compute_tile`** (~L1284)：ks 循环前一次性
   `b_scales_kpair = <K-pairing 读取>(bs_buf, bs_bases[0])`，传给所有
   `_a_streaming_compute`。

4. **`_load_b_and_scales`** (~L1114)：`b_kpair` 时 `b_scales_all = b_scales_kpair`
   （闭包复用，不发 ds_load）；`b_frags(ks)`、`a_scales(ks)` 照常。

5. **`_emit_wmma`** (~L1150)：加 `ks` 形参；`b_kpair` 时 `b_scale_idx = wn`、
   `scaleAType = ks`。删除 a8w4 的 `b_opsel_on`/per-tile 分支。
   `_a_streaming_compute._emit_rows` 把 `ks` 透传进来。

6. **计数 / 调度**（正确性关键，GPU 验）
   - `_b_scale_ds_loads_full`：`b_kpair` = `wmma_n_rep`（每 N-tile 一条，tile 开头一次）。
   - `_bs_ds_loads`（partial-drain wait）：`b_kpair` **去掉 b-scale 项**
     （`wmma_n_rep*_b_frag_loads_per_wn + (wmma_m_rep+3)//4`）。
   - `hot_loop_scheduler`：b-scale 的 `sched_dsrd` hint 从「每 ks」移到「tile 开头一次」。

7. **删除 a8w4 op_sel 开关 / Remove a8w4 op_sel toggle**
   - `weight_scale_opsel` kernel 参数、`WEIGHT_SCALE_OP_SEL` env、JIT cache key、
     `_GroupedA8W4Config` 字段及 masked compiler 透传
     （`grouped_moe_gfx1250.py`、`moe_grouped_gemm_mxscale_gfx1250.py`）。
   - `b_opsel_on` 中 weight 相关部分；a8w4 的 `_b_n4k8_pair`。
   - **保留** `use_scale_opsel` 对**激活 scale**(a_opsel) 的作用（与 weight 无关）。

## 验证 / Validation
每次 GPU 跑前 `rocm-smi --showpids` 确认空闲；改 kernel 后 `rm -rf /root/.flydsl/cache`。
1. **数值**（varied weight scale，`logits_diff < 0.01` gate，×3 确定性）：
   - 小 K：a8w4 swiglu E8/T8/512x512（2 k-tiles，无 steady-state）。
   - 大 K：E256/T8/topk8/7168x2048、E128/T8/topk8/3072x3072。
   - 期望与现有 per-tile/per-pair 读法数值一致（≈0.005），且 ×3 完全相同。
2. **ISA dump**（`FLYDSL_DUMP_IR=1`）：应出现「同一 acc、scaleA 同一寄存器、
   op_sel ROW0/ROW1、**SrcB 不同（不同 K-step）**」——即 K-pairing。
3. **不回归**：a4w4/fp4、fp8 数值与之前一致。

## 风险 / Risks
- wait / `sched_dsrd` 计数调错 → race 或读到未就绪 scale。用 ×3 确定性 + ISA 核对。
- multi-buffer：`b_scales_kpair` 在每个 `compute_tile`（对应一个 buffer/k-tile）内部
  加载，天然「每 buffer 一次」，无需额外处理。

## 落点 / Where
当前分支 `n4k8-squash` 上实现（与 logits_diff gate 同处），验证通过后再决定去留。
