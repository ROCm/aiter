# Dispatch + GEMM1 融合 (Phase 1a-v2 / fp8 transport) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## ⚠️ 路线修正（2026-07-29 实测后）—— 本 plan 现行路线为 1a-v2

原 1a（把 per-token bf16→fp8 量化折进 dispatch 接收路径、双写 `disp_out_q`/`disp_out_qscale`）**已实测为负优化并废弃**：它在照常 P2P 传 bf16 之上又加了一份 fp8 跨 rank 写，`ep_dispatch` +572us、整层 +18%（详见 spec §8.5）。对照 DeepGEMM（spec §8.6）：量化在 dispatch **之前**、**传输即 fp8**、dispatch 是**纯 fp8 mover**。

**现行路线 = 1a-v2（fp8 transport，spec §8.7）**：让 fp8 **取代** bf16 上线，而非并存。

- **T-A（打底，收益为正、可测）**：dispatch 前把层输入量化成 fp8+e8m0（每源 token 一次）→ dispatch 用 `data_type=fp8`+`scale_dim=hidden//32` **只发 fp8+e8m0（传输减半）** → GEMM1 a1 prep 消费 fp8 `recv_x`+`out_scales`，**复用已落地的 fp8-gather**（`build_moe_fp8_gather_preshuffle_route_ksplit_module` / `flydsl_moe_fused_quant_preshuffle(in_fp8_payload=,in_fp8_scale=)`）。
- **T-B（真·dispatch+GEMM1 融合，用户目标的收益点）**：把 a1 的 gather+preshuffle **折进 GEMM1 的 A-load prologue**，消除独立 a1 kernel launch + grouped a1 的一整趟 HBM 往返（写 `contiguous_m×hidden` fp8 再读回）。改动集中在 `batched_gemm_mxfp4` 的 TDM a8w4 A-loader。
- **T-C**：端到端 `MEGA-CHECK PASS` + profile A/B（dispatch 字节减半、a1 kernel 消失）。
- **终极**：Phase 3 整层巨核（spec §7），dispatch warps 直喂 GEMM1 ring。

**复用产物（已落地并验证）**：`_emit_fp8_gather_preshuffle` + fp8-gather route-ksplit builder + `flydsl_moe_fused_quant_preshuffle` 的 fp8-in 分支（原 Task 3）。**弃用**：`emit_per_token_mx_quant` 在 `make_dispatch` 内的调用 + `disp_out_q`/`disp_out_qscale` region（`AITER_EP_FUSE_DISPATCH_GEMM1`，默认关，保留作负结果实证）。

> 下方"## Global Constraints"及"## Task 1–4"为**已废弃的原 1a 设计**，保留作历史与对照，勿据其实现；实现请依下方 **1a-v2 Tasks（T-A/T-B/T-C）**。

---

## 1a-v2 Tasks（现行路线，bite-sized）

### 复用/事实（实现前必读，勿再造轮子）

- **dispatch 已支持 fp8 传输**：`EpDispatchCombineConfig(data_type=fp8, scale_dim=hidden//32, scale_type_size=1)` ⇒ `is_fp8=True`，`op.dispatch(x_fp8, wts, scale_e8m0, ids)` 会把 per-token fp8 payload + per-token e8m0 scale 一并 P2P 到目标 rank，接收端产出 `recv_x`(fp8) + `out_scales`(e8m0，到达序)。**不是新写代码，是启用现有分支**。
- **a1 fp8-gather 已落地并验证**（原 Task 3）：`build_moe_fp8_gather_preshuffle_route_ksplit_module` / `flydsl_moe_fused_quant_preshuffle(in_fp8_payload=, in_fp8_scale=)`——输入 per-token fp8+e8m0，输出 grouped 预混洗 `a1_payload`/`a1_scale`，直接喂 `flydsl_grouped_gemm_a8w4_masked`。
- **gemm1 消费点**：`grouped_moe_gfx1250.py:602/615` `flydsl_grouped_gemm_a8w4_masked(out, a1_payload, w1_u8, a1_scale, w1s_i32, psum, ...)`。
- **gemm 已有 gather 钩子先例**：`flydsl_grouped_gemm_a8w4_masked` 已有 `ep_tdm_gather` + `ep_rowmap`（gemm2→combine 侧 P2P store 用），T-B 在 A-load 侧对称新增"按 rowmap gather A"。
- **量化数值基准**：`_quantize_mxfp8_payload` / `per_token_mx_fp8` 参考实现（`grouped_moe_gfx1250.py` 内）——T-A 的 dispatch 前量化必须与它 byte 级一致（e4m3 + RoundUp e8m0，1×32 block）。

### Task T-A：fp8 transport 打底（收益为正、可测）

**目标**：dispatch 只传 fp8+e8m0（传输减半），a1 复用 fp8-gather，数值与基线等价（同样单次量化，只是提前到 dispatch 前）。**不改 kernel 边界**。

**Files**：`op_tests/multigpu_tests/test_mega_moe.py`（`_layer_step`）、`aiter/ops/flydsl/grouped_moe_gfx1250.py`（`_grouped_a8w4_tdm_moe` 入口）、EP dispatch config 构造处。

- [x] **T-A.1 传输往返单测（红→绿）**：`op_tests/flydsl_tests/test_fp8_transport.py::test_fp8_transport_roundtrip_byte_exact`——单 rank，`data_type=torch.float8_e4m3fn, scale_dim=H//32, scale_type_size=1`；`per_token_mx_fp8` 预量化 → dispatch → 用 `handle.disp_tok_id_to_src_tok_id_local`（recv-slot→src）反查，断言 `recv_x` fp8 payload 与 `out_scales` e8m0 **逐字节**对齐（`atol=0,rtol=0`）。**PASS**。锁定：dispatch 是纯 fp8+e8m0 byte-mover，scale 走 `out_scales`(packed i32→view uint8 取前 H//32)。
- [x] **T-A.2 dispatch 前量化 helper**：**选型 = 复用 `aiter.ops.triton.quant.dynamic_mxfp8_quant`**（即 moe a1 naive 路径 `_quantize_mxfp8_payload` 内部用的同一 per-1x32 MXFP8 量化，产 `(y_fp8[T,H], e8m0[T,H//32] uint8)`）。理由：与 GEMM1 fp8-gather 消费布局**天然同源**，dispatch 传的就是 GEMM1 输入字节，下游不 requant。helper `quantize_mxfp8_for_dispatch` 落在 `test_fp8_transport.py`。**判据修正**：`dynamic_mxfp8_quant` 的 e8m0 舍入与 RoundUp 参考在 ~1% block 差 1（约定差异，非 bug；因 fp8-gather 逐字节透传、不 requant，只需 pair 自洽）。故断言从"字节匹配"改为**反量化还原度**：`test_dispatch_quant_helper_roundtrips` 断言 `(payload,e8m0)` 反量化回 x 的 mean rel err <3%、max <20%（fp8-e4m3 3 尾数位）。**PASS**。
- [x] **T-A.3 a1 消费 fp8 recv**：**代码**——`_grouped_a8w4_tdm_moe` 的 `_use_disp_q` 分支（Task-3 已建）新增 uint8 coercion：`recv_x`(fp8, 1B/elem) 与 `out_scales`(e8m0 bytes) 非 uint8 时 `.view(torch.uint8)`（零拷贝），再 reshape 喂 fp8-gather，取代 `:575` 的 bf16 量化。**parity 单测** = `test_dispatch_quant_helper_byte_exact_vs_bf16_a1`：上游 helper 量化 → fp8-gather+preshuffle 的 `a1_payload/a1_scale` 与 baseline bf16 量化+gather+preshuffle **逐字节相等**（`atol=0,rtol=0`），即该分支实际发起的 kernel 调用。**PASS**。**全栈 fp8-hidden 接受性**：代码走查确认无阻塞（`M,topk`←`topk_ids`、`model_dim/inter_dim`←权重、`dtype` assert 仅针对输出 bf16、`_use_disp_q` 不读 hidden 做量化），live 实证并入 T-A.4。
- [x] **T-A.4 e2e 接线 + A/B**：**代码**——`DeviceMoEPipeline.setup` 加 `AITER_EP_FP8_TRANSPORT` 门控（默认关）：开时用 `dispatch_data_type=fp8, combine_data_type=bf16, scale_dim=H//32, combine_mode=gather`（**约束发现**：op 拒绝 fp8+scatter，非对称 dtype 仅 gather，故 scatter_fused 延后）。`_layer_step` fp8 分支：`quantize_mxfp8_for_dispatch(xn)` → `dispatch(x_fp8, wts, e8m0, ids)` → `recv_x/out_scales` 经 `ep_disp_q_payload/scale` 透传给 moe。
  - **正确性**：2-rank 与 4-rank `MEGA-CHECK PASS`，**logits_diff 与 baseline 逐位相同**（2r:0.002060、4r:0.003119）——证实 byte-exact e2e 等价（flydsl 量化选型正确）。
  - **A/B profile (hd=7168, 2-rank)**：`ep_dispatch` **986.3→776.4us（-210us/-21%）**——fp8 传输带宽减半的收益**真实可测**，无 1a 双写回归。但新增独立上游量化 pass（215.6us over ct 全宽 bf16 读）+ a1 fp8-gather(89.5us) 拆散了 baseline 的融合 a1(78.5us)，故**整层 total 近似持平**（1091.7→1097.5us）。
  - **结论**：T-A 打底达成（fp8 transport 正确 + dispatch 带宽 -21% 实测），但净收益被"独立量化 pass"抵消。**净正收益需 T-B**（消除 a1 gather + grouped-a1 HBM 往返）与/或把上游量化融进 `_rmsnorm`。

- [x] **T-A.4b rmsnorm+量化融合（净转正）**：把 `_rmsnorm`(torch) + 独立 flydsl 量化(216us) 两趟替换成单个 `aiter.ops.triton.quant.fused_rms_mxfp8_quant(x, ones, eps)`（`helper rmsnorm_mxfp8_for_dispatch`，一趟读 x 直出 fp8+e8m0；ones weight 匹配无 gain 的 `_rmsnorm`）。仅在 `sw1 is None`（无 shared FFN 需 bf16 xn）时启用，否则回退 `_rmsnorm`+`quantize_mxfp8_for_dispatch`。
  - **正确性**：2/4-rank `MEGA-CHECK PASS`，logits_diff 0.002133/0.003176（Triton rmsnorm+量化约定与 byte-exact 微差，远在 tol=0.1 内）。
  - **A/B (hd=7168, 2-rank, 复现×2)**：独立量化 216us → `_fused_rms_mxfp8_kernel` **30.5us**；`ep_dispatch` 保持 ~797us（-21%）。**整层 total baseline ~1090us → fp8+rmsfuse ~873us，稳定 -217us (-20%)**。
  - **结论**：**T-A 净收益转正（-20% 整层）**，即用户要的"可测收益"。dispatch 带宽减半的收益经 rmsnorm 融合流到底线；T-B 在此之上继续消除 a1 gather + grouped-a1 HBM 往返。

**约束**：保留 `topids_to_rows`/`contiguous_psum_remap`/`ep_rowmap` 与 gemm2+combine `scatter_fused` 融合零改动；只把传输 dtype 从 bf16 换成 fp8；`AITER_EP_FP8_TRANSPORT=0`（默认）必须与主线逐 kernel 一致。

### Task T-B：真·dispatch+GEMM1 融合（消除 a1 kernel + grouped a1 HBM 往返）

**目标**：gemm1 的 A-tile 直接从 fp8 `recv_x` **按 route map gather**（每行 = 连续 K 向量，行间用 rowmap 重定基址）并**内联** e8m0 预混洗，取消独立 a1 fp8-gather kernel launch 与 `grouped_a1`(`E×max_m×hidden` fp8) 的写出+读回一整趟 HBM。前置：T-A 已绿。

**Files**：`aiter/ops/flydsl/batched_gemm_mxfp4.py`（`flydsl_grouped_gemm_a8w4_masked` 及其 kernel builder 的 A-load / scale(SFA)-load 发射处）、`aiter/ops/flydsl/grouped_moe_gfx1250.py`（`_grouped_a8w4_tdm_moe` gemm1 调用处）。

- [x] **T-B.1 定位 A-load / scale-load 发射点（探查完成，feasibility verdict）**：
  - **A-tile 载入**：`gemm_mxscale_gfx1250.py:928-947`（`make_desc_a`）+ `:3056-3068`（`issue_tdm_loads`）——**单次连续 TDM 2D 矩形块** `[tile_m, packed_tile_k]`，行基址 `flat_m_base_input`（连续 grouped rows），行 stride `K_packed_a`。grouped 行由 psum(`arg_m_tile_map`) bisect 出 expert（contiguous: `:3900-4011`，`layout_row=flat_m_tile*tile_m` 作 `flat_m_base_override`）。**非逐行寻址**。
  - **scale(SFA) 载入**：`:970-997`（`make_desc_as`）——同样单块 TDM，`a_scale_row_base = flat_m_base // wmma_m_rep`，**假定 A 行连续**。
  - **ep_rowmap 先例**：仅在 **GEMM2 输出 scatter** 侧（`:1877-1957` epilogue P2P，`moe_contiguous_psum.py:830-843` host 构建）；A 输入侧**无** gather/rowmap。
  - **⚠️ Feasibility verdict**：B1（把 gather 折进 A-load）与现有"单块 TDM + wave-specialized 4-stream 流水线"架构**根本冲突**。要 gather 分散 `recv_x` 行需 (a) 逐 WMMA 行发独立 TDM（破坏流水线/coalescing）或 (b) 新建 load 侧 TDM gather（大工程，scale 须按同一 rowmap 同步 gather，不能复用单块 `make_desc_as`）。当前架构本就靠"GEMM 前 pre-gather 到 contiguous"（即 a1 kernel）规避此问题。收益面：a1 gather ~89us + grouped-a1(`contiguous_m×hidden` fp8) 一趟 HBM 往返；成本面：高风险大重写。**结论：B1 非 bite-sized，需重新决策（见下）**。
- [x] **T-B.2 确认活跃 builder + load 侧 gather 原语**：活跃路径 = **TDM batched**（`mxfp4_preshuffle_gfx1250_tdm.py`，`_use_a8w4_tdm_path/_ep` 默认 ON）。**load 侧 gather 原语已备**：`tdm_gather_shim.py::tensor_load_gather` + `make_tensor_gather_descriptor`（行 i 地址 = base + rowidx[i]*stride，32b≤8 索引/条），已用于 store 侧 `ep_tdm_gather`。A payload 载入 = `:290 add_tdm_loads(gA_base, blk_m*A_KROW, ...)` 连续块；scale 载入 = `:294` preshuffled 布局。
- [x] **T-B.3 实现 GEMM1 A-gather（kernel/host/caller，identity-gather parity 绿）**：`launch_gemm_a8w4_tdm` 加 `ep_a_gather`+`arg_a_rowmap`+`i32_num_recv_rows`。A-load 分支用 `tensor_load_gather`：每 wave 领 `g % num_waves == wave` 的组（8 行/32b-index 条），per-k-tile `global_byte_off = kt*A_ROW_B`，写进 **packed** A LDS（`LDS_PAD_A=0` ⇒ `A_LDS_ROW==A_ROW_B`，匹配 `load_a` 索引）；`r>=mn_oob` 或 rowmap<0 用 OOB 索引（HW drop）。pipeline outstanding：`TDM_PER = gathers_per_wave + 3`（B/SA/SB），**wave-spec 先关**求每 wave 均匀计数，要求 `ceil(tile_m/8) % num_waves == 0`。scale 仍走现成路径（只 gather 大头 payload）。踩坑修复：runtime `tensor_dim1` 需 `_to_raw` 成 raw arith；`kt` 是 scf.for 归纳变量 ⇒ `global_byte_off` 必须 runtime `index_cast`。**parity 单测** `test_tb_a_gather_parity::test_gemm1_identity_gather_matches_contiguous`：a8w4 gugu no-bias moe，identity gather（源=已分组 a1_payload，rowmap=arange）输出与 contiguous load **逐字节相等**——隔离验证 gather 机制。**PASS**。默认路径（gather off）不变。
- [x] **T-B.4 真 rowmap + 单 GPU parity**：确认 `contiguous_psum_remap` 把 `topids_to_rows` **原地**重映射到 contiguous 空间；a1 物化用 `E=1,max_m=contiguous_m,source_topk=topk` ⇒ `source_row = route//topk`。共享 helper `_build_a_gather_rowmap(topids_to_rows, contiguous_m, topk, num_recv_rows, num_valid_routes)`：`a_rowmap[contiguous_row] = route//topk`（图安全静态形状，invalid/dead-tail 走 dump 槽 + OOB 索引）。`_grouped_a8w4_tdm_moe` 的 `_use_disp_q` 分支：`_gemm1_a = _dq_p.reshape(-1,model_dim)`（**未分组 recv**）+ 真 rowmap 调 gemm，**不再消费 a1_payload**。**parity 单测** `test_real_rowmap_matches_materialized_a1`：用 `flydsl_moe_topids_to_rows`+`contiguous_psum_remap` 建同款 contiguous t2r，断言 `materialized_a1[g] == src[a_rowmap[g]]`（每个 valid grouped 行逐字节）。**PASS**。⇒ 机制正确(T-B.3) ∧ rowmap 正确(T-B.4) ⇒ 真 gather 与消费物化 a1 逐字节等价。
- [x] **T-B.5 EP mega e2e + 收益证明 → 负结果，A-load gather 废弃**（2026-07-29，4×gfx1250，e384/k6/hd7168/id3072/scatter_fused/layers=2）：`AITER_EP_FP8_TRANSPORT=1 AITER_EP_A_GATHER=1` 端到端**数值正确**（`MEGA-CHECK PASS`，logits_diff=0.002158，与纯 fp8 transport 逐位一致），但**性能严重负优化**：gemm1 `...K7168` per_call **177.6→639.0us（+3.6×）**、整层 wall **538.0→1098.5us（+2.0×）**、device/layer 547.5→1053.7us。**根因**（= T-B.1 feasibility verdict 落地）：(1) `mxfp4_preshuffle_gfx1250_tdm.py:288` `WAVE_SPEC = ... and not ep_a_gather` ⇒ 一开 gather 就关掉 4-stream wave-spec 流水线；(2) A-load 从单块连续 TDM 退化为逐组 `tensor_load_gather`，coalescing 崩塌。即使拆成 scale-only 省掉 a1_payload 写（~53us/layer + 一趟 fp8 HBM 写），也远补不回 gemm1 的 +461us/call。**结论**：T-B（把 gather 折进 GEMM1 A-load）在当前"单块 TDM + wave-specialized"架构下走不通，**废弃**（`AITER_EP_A_GATHER` 开关保留、默认关，作负结果实证；机制/rowmap 单测 T-B.3/B.4 仍绿，留作 Phase 2/3 dispatch warp→gemm ring 组件参考）。详见 spec §8.8。**后续 ROI 排序**：① 修 combine 非对称 dtype 限制（恢复 fp8-transport 下 `scatter_fused` 融合 combine，+66us/call 实打实回退）② T-B fallback（dispatch grid 内第二遍拷 grouped a1）③ Phase 3 整层巨核（gather 移到 ring 生产侧）。
- [ ] **T-B fallback（若 gather-load 引入 coalescing 回退）**：退化为"在 dispatch 现有 grid 内做第二遍把到达序 fp8 拷成 grouped 连续 a1"（省 kernel launch，不省 HBM 往返），或直接推进 Phase 3 整层巨核（spec §7）。二选一并记录实测依据。

### Task T-C：端到端正确性 + 性能验收

- [ ] **T-C.1 正确性**：2 rank 与 4 rank 均 `MEGA-CHECK PASS`，logits 与基线（`AITER_EP_FP8_TRANSPORT=0`）同量级。
- [ ] **T-C.2 性能 A/B 表**：基线 vs T-A vs T-B，列 `ep_dispatch`、a1 prep kernel、gemm1、整层 device time；确认 (1) T-A dispatch 传输减半且无 1a 双写回归，(2) T-B a1 kernel 消失 + 少一趟 HBM 往返。写回 spec §8.7 结果区。
- [ ] **T-C.3 回归门**：`AITER_EP_FP8_TRANSPORT=0` 默认路径与主线逐 kernel 一致（CI/手测）。

---

**Goal:** 在默认的 a8w4 TDM contiguous 路径上，把"bf16→fp8 量化 + e8m0 scale"折进 dispatch 接收路径，让 gemm1 的输入准备不再对 `hidden_states` 做 bf16 全宽读回。分两个增量：**1a**（安全，先落地）把 per-token 量化搬进 dispatch，把 `_grouped_a8w4_tdm_moe:555` 的 a1 `flydsl_moe_fused_quant_preshuffle` 降级为"读已量化 fp8 + gather + preshuffle"（省 bf16 全宽读 + 省量化算）；**1b**（后续里程碑）让 dispatch 接管 grouped 行分配 + 路由图，彻底消除该 kernel launch。

**Architecture:** 不改 kernel 边界（persistent 单核是 Phase 2/3）。保留 `topids_to_rows` / `contiguous_psum_remap`（⇒ `ep_rowmap` 与 gemm2+combine `scatter_fused` 融合零改动、留在默认 TDM 路径）。1a 只改：dispatch 接收 epilogue 增写 per-token fp8 payload + per-token e8m0 scale（**到达序**，与 `disp_out` 同索引，无需前缀和、顺序天然一致）；新增/改造 a1 准备 kernel 从 fp8 读入做 gather+preshuffle。整链由 env `AITER_EP_FUSE_DISPATCH_GEMM1` 门控，默认关。

**Tech Stack:** Python 3, PyTorch, aiter FlyDSL (flyc.jit kernels), cco 对称内存 (`SymmArena` / `Window.lsa_ptr`), gfx1250 WMMA a8w4 (fp8 activation + MXFP4 weight)。

## Global Constraints

- 目标硬件仅 gfx1250；其它硬件走原路径不受影响。
- 默认关闭，仅 `AITER_EP_FUSE_DISPATCH_GEMM1=1` 启用；`=0`（默认）必须与当前主线逐 kernel 一致。
- 只作用于 a8w4（`data_format=="a8w4"`，`quant_mode` 固定 `"fp8"`）；fp4 数据路径不启用。
- 与 `--combine scatter_fused`（默认 `AITER_EP_SCATTER_TDM=1`）正交，可组合。
- **顺序一致性铁律**：1a 中 dispatch 只按**到达序**（`dest_tok_id`，与 `disp_out` 同）写 per-token 量化产物；grouped 行分配仍由 `topids_to_rows`+`contiguous_psum_remap` 决定，dispatch 不得改变行分配。
- config 事实：字段是 `num_experts_per_rank`（非 `num_experts//world_size`）；`is_fp8/is_fp4` 指传输 dtype，a8w4 传 bf16，故不能用它判量化目标——量化目标固定 fp8。
- CUDA graph 兼容：固定 kernel 序列，无 host 端动态分配。
- 测试入口：`op_tests/multigpu_tests/test_mega_moe.py`，非 `/app` 目录运行（避免 `/app/triton` 遮蔽）。判据：`--acc_verify 1` 打印 `MEGA-CHECK PASS`。

---

## Task 1: env 开关 + config 属性 + per-token 量化 arena region

**Files:**
- Modify: `aiter/aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py`
  - `EpDispatchCombineConfig` 属性区（`235..396` 附近，与 `is_fused` `239..243` 同风格）
  - `EpDispatchCombineOp.__init__` regions 列表（`478..492`）
- Test: `aiter/op_tests/flydsl/test_dispatch_gemm1_fusion.py`（新建）

**Interfaces:**
- Consumes: 现有 `EpDispatchCombineConfig`（`num_experts_per_rank`, `world_size`, `hidden_dim`, `effective_max_recv`）。
- Produces（供 Task 2/3）:
  - `EpDispatchCombineConfig.fuse_dispatch_gemm1: bool`
  - arena regions（仅开关开时追加）:
    - `"disp_out_q"`：`effective_max_recv * hidden_dim` 字节（uint8，per-token fp8 payload，到达序）
    - `"disp_out_qscale"`：`effective_max_recv * (hidden_dim // 32)` 字节（uint8，per-token e8m0，**未** preshuffle）
  - `EpDispatchCombineOp.disp_out_q_view() -> (payload, scale)`：本地 tensor，形状 `(recv_cap, hidden_dim)` uint8 与 `(recv_cap, hidden_dim//32)` uint8
  - 模块级纯函数 `_fused_q_regions(cfg) -> list[(name, nbytes)]`（关时返回 `[]`）

- [ ] **Step 1: 写失败测试 —— 开关与 region 纯函数**

新建 `aiter/op_tests/flydsl/test_dispatch_gemm1_fusion.py`：

```python
import torch
from aiter.ops.flydsl.dispatch_combine_v2.dispatch_combine_op import (
    EpDispatchCombineConfig,
    _fused_q_regions,
)


def _make_cfg(**ov):
    base = dict(
        rank=0,
        world_size=2,
        hidden_dim=512,
        max_num_inp_token_per_rank=128,
        num_experts_per_rank=4,
        num_experts_per_token=2,
        data_type=torch.bfloat16,
    )
    base.update(ov)
    return EpDispatchCombineConfig(**base)


def test_fuse_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("AITER_EP_FUSE_DISPATCH_GEMM1", raising=False)
    assert _make_cfg().fuse_dispatch_gemm1 is False


def test_fuse_flag_on(monkeypatch):
    monkeypatch.setenv("AITER_EP_FUSE_DISPATCH_GEMM1", "1")
    assert _make_cfg().fuse_dispatch_gemm1 is True


def test_q_regions_present_when_on(monkeypatch):
    monkeypatch.setenv("AITER_EP_FUSE_DISPATCH_GEMM1", "1")
    cfg = _make_cfg()
    regs = dict(_fused_q_regions(cfg))
    assert regs["disp_out_q"] == cfg.effective_max_recv * cfg.hidden_dim
    assert regs["disp_out_qscale"] == cfg.effective_max_recv * (cfg.hidden_dim // 32)


def test_q_regions_absent_when_off(monkeypatch):
    monkeypatch.delenv("AITER_EP_FUSE_DISPATCH_GEMM1", raising=False)
    assert _fused_q_regions(_make_cfg()) == []
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl/test_dispatch_gemm1_fusion.py -v`
Expected: FAIL —— `ImportError: cannot import name '_fused_q_regions'`

- [ ] **Step 3: 实现 config 属性 + region 纯函数**

`EpDispatchCombineConfig` 属性区（`is_fused` 后）加：

```python
    @property
    def fuse_dispatch_gemm1(self) -> bool:
        # a8w4 gfx1250: fold per-token bf16->fp8 quant into dispatch recv path.
        return os.environ.get("AITER_EP_FUSE_DISPATCH_GEMM1", "0") in (
            "1", "true", "True", "yes", "on",
        )
```

模块级（`_align_up` 附近）加：

```python
def _fused_q_regions(cfg):
    """Per-token fp8 payload + e8m0 scale regions (arrival order), added only
    when dispatch->gemm1 quant fusion is enabled. Empty otherwise so the default
    arena layout is byte-for-byte unchanged."""
    if not cfg.fuse_dispatch_gemm1:
        return []
    cap = cfg.effective_max_recv
    h = cfg.hidden_dim
    return [
        ("disp_out_q", cap * h),            # uint8 fp8 payload, [recv_cap, hidden]
        ("disp_out_qscale", cap * (h // 32)),  # uint8 e8m0, [recv_cap, hidden//32]
    ]
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl/test_dispatch_gemm1_fusion.py -v`
Expected: PASS（4 个）

- [ ] **Step 5: 接进 `__init__` + view helper**

regions 列表构造后（`dispatch_combine_op.py:490` 之后、`self.arena = SymmArena(...)` 之前）加：

```python
        regions += _fused_q_regions(cfg)
```

在 op 的 view 方法区（`disp_out` 相关，`769` 附近）加：

```python
    def disp_out_q_view(self):
        cfg = self.cfg
        cap, h = self._recv_cap, cfg.hidden_dim
        payload = from_gpu_ptr(self.arena.local_ptr("disp_out_q"), (cap, h), torch.uint8)
        scale = from_gpu_ptr(
            self.arena.local_ptr("disp_out_qscale"), (cap, h // 32), torch.uint8
        )
        return payload, scale
```

（`from_gpu_ptr` 已在文件顶部从 `mori.tensor_utils` 导入，见 `dispatch_combine_op.py:30`。）

- [ ] **Step 6: 运行确认仍通过 + 提交**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl/test_dispatch_gemm1_fusion.py -v`
Expected: PASS

```bash
cd /app/aiter
git add aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py op_tests/flydsl/test_dispatch_gemm1_fusion.py
git commit -m "feat(ep): dispatch->gemm1 fusion flag + per-token fp8 quant arena regions"
```

---

## Task 2: dispatch 接收 epilogue —— per-token bf16→fp8 量化 (到达序)

在 `make_dispatch` 的 token-embedding scatter 之外（或之中）增写：每个 recv token（`dest_tok_id`）除了照常把 bf16 写进 `disp_out`，额外把它 fp8 量化 + 每 32 元素 e8m0，写进 dest peer 的 `disp_out_q[dest_tok_id]` 与 `disp_out_qscale[dest_tok_id]`。**到达序、逐 token、无 preshuffle**，因此与 `disp_out` 索引一致，无需前缀和。

**Files:**
- Modify: `aiter/aiter/ops/flydsl/dispatch_combine_v2/intranode_kernels.py`（`make_dispatch`：`run(...)` 签名 `352..` 与 token scatter 主体 `263..297`）
- Modify: `aiter/aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py`（dispatch launch 参数组装 `604..614`）
- Reference（复刻量化数学，不改）: `aiter/aiter/ops/flydsl/moe_kernels.py` `_get_compiled_fused_quant_preshuffle` 的 bf16→fp8 + e8m0 device body（**只取 quant 部分，不取 preshuffle**）
- Test: `aiter/op_tests/flydsl/test_dispatch_gemm1_fusion_kernel.py`（新建，单 rank 数值对照）

**Interfaces:**
- Consumes: Task 1 的 `disp_out_q` / `disp_out_qscale` offsets；`hidden_dim`。
- Produces: `make_dispatch(..., fuse_quant: bool, off_disp_out_q, off_disp_out_qscale, hidden_dim)`；运行后 `disp_out_q[t]`/`disp_out_qscale[t]` == 对 `disp_out[t]`（bf16）做 fp8 量化+e8m0 的结果（逐 token，未 preshuffle）。

- [ ] **Step 1: 写失败测试 —— per-token 量化对齐参考**

单 rank 即可验证。参考值：对 dispatch 后的 `disp_out`（bf16, `[recv, hidden]`）逐 token 跑与 `_get_compiled_fused_quant_preshuffle` 相同的 fp8+e8m0 量化（可用一段 numpy/torch 参考实现，或复用 kernel 的 fp8/e8m0 编码函数）。

新建 `aiter/op_tests/flydsl/test_dispatch_gemm1_fusion_kernel.py`：

```python
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs gfx1250")


def _ref_fp8_e8m0(disp_bf16):
    # Mirror moe_kernels fp8 payload + per-32 e8m0 scale encoding.
    # (fill with the exact rounding/scale formula copied from
    #  _get_compiled_fused_quant_preshuffle; MX block = 32 elems.)
    ...


def test_dispatch_per_token_quant_matches_ref(monkeypatch):
    monkeypatch.setenv("AITER_EP_FUSE_DISPATCH_GEMM1", "1")
    from aiter.op_tests.flydsl._fusion_harness import run_single_rank_dispatch

    disp_out, payload, scale, n_recv = run_single_rank_dispatch(
        num_experts_per_rank=4, hidden=512, ntok=64, topk=2
    )
    ref_payload, ref_scale = _ref_fp8_e8m0(disp_out[:n_recv])
    torch.testing.assert_close(payload[:n_recv], ref_payload, atol=0, rtol=0)
    torch.testing.assert_close(scale[:n_recv], ref_scale, atol=0, rtol=0)
```

新建 `aiter/op_tests/flydsl/_fusion_harness.py`，`run_single_rank_dispatch` 构造单 rank comm + `EpDispatchCombineOp`，跑 `dispatch()`，返回 `disp_out`（bf16 view）、`disp_out_q_view()` 的 `(payload, scale)`、以及有效 recv 数 `n_recv`。comm bootstrap 从 `test_mega_moe.py` 的单 rank 初始化抽取（按其真实 API 填充）。

- [ ] **Step 2: 运行确认失败**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl/test_dispatch_gemm1_fusion_kernel.py -v`
Expected: FAIL —— `disp_out_q` 全 0（融合分支未接）。

- [ ] **Step 3: kernel 增写 per-token 量化**

`run(...)` 签名（`intranode_kernels.py:352`）追加 `const_expr` `fuse_quant` + 指针 `off_disp_out_q`, `off_disp_out_qscale` + `hidden_dim`。

在 token scatter 块内（`263..297`，已把该 token 的 bf16 从 `local_tok_addr` 拷到 dest peer `off_out_tok`），追加（仅 `const_expr(fuse_quant)` 且 `do_publish` 且非 dup/overflow）：

```python
if const_expr(fuse_quant):
    # per-token bf16 -> fp8 + per-32 e8m0, arrival-order slot = dest_tok_id.
    # Mirror the quant math (not the preshuffle) from moe_kernels
    # _get_compiled_fused_quant_preshuffle.
    peer_q = fx.Int64(window.lsa_ptr(dest_pe, off_disp_out_q))
    peer_qs = fx.Int64(window.lsa_ptr(dest_pe, off_disp_out_qscale))
    # for each 32-elem MX block owned by this lane:
    #   load 32 bf16 from local_tok_addr, compute e8m0 scale + fp8 payload,
    #   store 32 fp8 bytes -> peer_q[dest_tok_id*hidden + blk*32 ...],
    #   store 1 e8m0 byte  -> peer_qs[dest_tok_id*(hidden//32) + blk].
```

要点：
- 复刻 `_get_compiled_fused_quant_preshuffle` 的 fp8 payload + e8m0 编码（同 rounding、同 scale），**只做量化、不做 preshuffle**（preshuffle 留给 Task 3 的 gather kernel）。
- 写入 slot 用 `dest_tok_id`（与现有 `off_out_tok` 同），保证到达序一致。
- lane 分工：沿用现有 token 拷贝的 lane 跨步，把 hidden 切成 32-elem MX block 分给 lane。

- [ ] **Step 4: host 侧接参数**

`dispatch_combine_op.py` dispatch launch 组装处（`604..614`）：

```python
        _fuse_q = cfg.fuse_dispatch_gemm1
        # ... existing off_* ...
        fuse_quant=_fuse_q,
        off_disp_out_q=arena.offset("disp_out_q") if _fuse_q else 0,
        off_disp_out_qscale=arena.offset("disp_out_qscale") if _fuse_q else 0,
        hidden_dim=cfg.hidden_dim,
```

- [ ] **Step 5: 运行确认通过 + 提交**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl/test_dispatch_gemm1_fusion_kernel.py -v`
Expected: PASS

```bash
cd /app/aiter
git add aiter/ops/flydsl/dispatch_combine_v2/intranode_kernels.py aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py op_tests/flydsl/test_dispatch_gemm1_fusion_kernel.py op_tests/flydsl/_fusion_harness.py
git commit -m "feat(ep): per-token bf16->fp8 quant in dispatch recv path (arrival order)"
```

---

## Task 3: a1 准备 kernel 从 fp8 读入 (gather + preshuffle)，替换 line-555 bf16 路径

**Files:**
- Modify: `aiter/aiter/ops/flydsl/moe_kernels.py`（`flydsl_moe_fused_quant_preshuffle` route 分支：新增 `in_is_fp8` 模式，输入已是 fp8 payload+e8m0，跳过量化，只 gather+preshuffle）
- Modify: `aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py`（`_grouped_a8w4_tdm_moe:555` 的 a1 调用）
- Test: `aiter/op_tests/flydsl/test_dispatch_gemm1_fusion.py`（追加）

**Interfaces:**
- Consumes: Task 2 的 `disp_out_q`/`disp_out_qscale`（到达序）；现有 `topids_to_rows`（route→源 token 行）、`row_starts`（contiguous psum）。
- Produces: `flydsl_moe_fused_quant_preshuffle(..., in_fp8_payload=..., in_fp8_scale=...)`：从 fp8 gather 到 contiguous grouped `a1_payload`/`a1_scale`（preshuffle），结果 == 现有 bf16 路径逐字节一致。

- [ ] **Step 1: 写失败测试 —— fp8 gather 路径 == bf16 路径**

在 `test_dispatch_gemm1_fusion.py` 追加（需 GPU）：先对随机 bf16 `hidden` 跑现有 `flydsl_moe_fused_quant_preshuffle`（bf16 in）得参考；再对同数据先做 per-token fp8 量化得 `(q, qs)`，走新 `in_fp8_payload` 分支，断言两者 `a1_payload/a1_scale` 逐字节一致。

```python
def test_fp8_gather_preshuffle_matches_bf16(monkeypatch):
    import torch
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle
    from aiter.op_tests.flydsl._fusion_harness import per_token_fp8_quant, make_route
    ...
    ref_p, ref_s = flydsl_moe_fused_quant_preshuffle(
        hidden.reshape(1, T, H), 1, cont_m, wmma_rep=wr, quant_mode="fp8",
        topids_to_rows=t2r, source_topk=topk,
    )
    q, qs = per_token_fp8_quant(hidden)  # same encoding as Task 2 kernel
    fp8_p, fp8_s = flydsl_moe_fused_quant_preshuffle(
        None, 1, cont_m, wmma_rep=wr, quant_mode="fp8",
        topids_to_rows=t2r, source_topk=topk,
        in_fp8_payload=q, in_fp8_scale=qs,
    )
    torch.testing.assert_close(fp8_p, ref_p, atol=0, rtol=0)
    torch.testing.assert_close(fp8_s, ref_s, atol=0, rtol=0)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl/test_dispatch_gemm1_fusion.py -k fp8_gather -v`
Expected: FAIL —— `flydsl_moe_fused_quant_preshuffle` 无 `in_fp8_payload` 参数。

- [ ] **Step 3: 给 `flydsl_moe_fused_quant_preshuffle` 加 fp8-in 模式**

在 host wrapper（`moe_kernels.py:2269`）加参数 `in_fp8_payload=None, in_fp8_scale=None`；当提供时：跳过 `assert grouped_in.dtype==bf16`，改走一个编译变体 `in_is_fp8=True`，device kernel 从 `in_fp8_payload`（`[T, H]` uint8）与 `in_fp8_scale`（`[T, H//32]` uint8）按 `topids_to_rows`（source row = route//source_topk）读入，**不做量化**，直接把 fp8 payload 复制 + 把 e8m0 scale 按 WMMA 4×32 preshuffle 索引（复刻 `_grouped_a8w4_preshuffle_e8m0_scale`）写进 grouped `out_payload/out_scale`。

- [ ] **Step 4: 运行确认通过**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl/test_dispatch_gemm1_fusion.py -k fp8_gather -v`
Expected: PASS

- [ ] **Step 5: `_grouped_a8w4_tdm_moe` 在开关下改喂 fp8**

`grouped_moe_gfx1250.py:555` 的 a1 调用包开关（`ep_op`/dispatch op 需能取到 —— 通过新增可选参数 `ep_disp_q=(payload, scale)` 传入，来源为 `EpDispatchCombineOp.disp_out_q_view()`）：

```python
    if ep_disp_q is not None:  # dispatch already produced per-token fp8
        _q, _qs = ep_disp_q
        a1_payload, a1_scale = flydsl_moe_fused_quant_preshuffle(
            None, 1, contiguous_m, wmma_rep=wmma_rep, quant_mode=_quant_mode,
            masked_m=None, topids_to_rows=topids_to_rows, source_topk=topk,
            num_valid_routes=_ep_nvr, in_fp8_payload=_q, in_fp8_scale=_qs,
        )
    else:
        a1_payload, a1_scale = flydsl_moe_fused_quant_preshuffle(
            hidden_states.reshape(1, token_num, model_dim), 1, contiguous_m,
            wmma_rep=wmma_rep, quant_mode=_quant_mode, masked_m=None,
            topids_to_rows=topids_to_rows, source_topk=topk, num_valid_routes=_ep_nvr,
        )
```

`ep_disp_q` 从调用链透传：`fused_moe`→…→`_grouped_a8w4_tdm_moe`。当 `cfg.fuse_dispatch_gemm1` 且 EP scatter 路径时，上层把 `ep_op.disp_out_q_view()` 传入。（透传参数名与上层函数以现网为准。）

- [ ] **Step 6: 运行 + 提交**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl/test_dispatch_gemm1_fusion.py -v`
Expected: PASS

```bash
cd /app/aiter
git add aiter/ops/flydsl/moe_kernels.py aiter/ops/flydsl/grouped_moe_gfx1250.py op_tests/flydsl/test_dispatch_gemm1_fusion.py
git commit -m "feat(ep): a1 prep reads dispatch fp8 (gather+preshuffle), skip bf16 read+quant"
```

---

## Task 4: 端到端正确性 + 性能对照

**Files:** Test: `op_tests/multigpu_tests/test_mega_moe.py`（复用）

- [ ] **Step 1: 正确性（融合 on）**

Run:
```bash
cd /tmp && AITER_EP_FUSE_DISPATCH_GEMM1=1 \
ENABLE_CK=0 AITER_FORCE_A8W4=1 AITER_USE_GROUPED_GEMM=1 AITER_BF16_FP8_MOE_BOUND=0 \
torchrun --standalone --nproc_per_node=4 \
  /app/aiter/op_tests/multigpu_tests/test_mega_moe.py \
  -q a8w4_mxfp4 -e 384 -k 6 -hd 7168 -id 3072 \
  --combine scatter_fused --layers 2 --acc_verify 1
```
Expected: `MEGA-CHECK PASS`

- [ ] **Step 2: 回归（融合 off）**

Run: 同上，`AITER_EP_FUSE_DISPATCH_GEMM1=0`。Expected: `MEGA-CHECK PASS`（默认路径不受影响）。

- [ ] **Step 3: 性能对照**

Run: on/off 各追加 `--profile_table 1 --layers 61`。
Expected: 融合 on 时 a1 `fused_quant_preshuffle` 的读入从 bf16(2B) 降为 fp8(1B)+scale、量化算力移除；per-layer 时间 ≤ 基线。若提升有限，记录到 spec §8 供 1b 决策（不阻塞正确性合入）。

- [ ] **Step 4: 记录结果并提交**

```bash
cd /app/aiter
git add docs/superpowers/specs/2026-07-29-dispatch-gemm1-fusion-design.md
git commit -m "docs(ep): record phase-1a dispatch fp8 fusion perf"
```

---

## 增量 1b（里程碑，本 plan 不展开为 bite-sized）: dispatch 接管 grouped 落位以消除 a1 kernel launch

- **前置:** 1a 落地（验证 in-dispatch fp8 量化数值正确）。
- **交付物:** dispatch 在接收结束后（grid-sync）计算 per-expert 前缀和，把 per-token fp8 payload/scale remap 到 **contiguous grouped 行**，并产出与之一致的 `topids_to_rows`/`psum`/`ep_rowmap`；`_grouped_a8w4_tdm_moe` 跳过 `topids_to_rows`+`contiguous_psum_remap`+a1 `fused_quant_preshuffle`，直接消费 dispatch 产出。
- **风险:** dispatch 必须精确复刻现 `contiguous_psum_remap` 的行分配与 `ep_rowmap` 语义，否则 gemm1/gemm2 读错行。需逐字段对齐。
- **收益:** 彻底消除 a1 kernel launch（1a 只降其成本）。
- **产出 detailed plan:** 待 1a 结果出来后再写。

---

## 后续 Phase（2 & 3）里程碑路线

（不变，见下节）Phase 2（dispatch+L1 单核）/ Phase 3（整层巨核）为框架级工作，依赖 FlyDSL 尚不具备的 kernel 内持久调度 / 生产-消费 barrier / 双级 GEMM 串联，详见 spec §4 / §7。此处仅列里程碑，落地前各补独立 detailed plan。

### 启动门槛
- Phase 2 启动：1a/1b 落地后 profile 显示瓶颈仍在 dispatch 通信本身。
- Phase 3 启动：Phase 2 落地后 profile 显示瓶颈仍在中间张量 HBM 往返 + 两端通信。

### Milestone P2-0: FlyDSL kernel 内持久调度框架（persistent 调度 / LDS 生产-消费 barrier / 动态 tile / grid+NVLink barrier）
### Milestone P2-1: per-expert recv-count 到齐信号（64-bit 低位计数 / 高位 SM×rank 上报数）
### Milestone P2-2: dispatch+L1 单 persistent kernel（复用 1a/1b 的 in-dispatch 量化作 prologue）
### Milestone P3-1: L2+combine 内联为巨核 epilogue（复用 scatter_fused），验证出口跨 rank barrier
### Milestone P3-2: 打通 L1→SwiGLU→L2 in-kernel 中转、去中间 HBM，含双级环形池死锁安全容量推导
### Milestone P3-3: 权重聚簇调度 + occupancy/寄存器调优

（每个里程碑：前置/交付物/验证 `MEGA-CHECK PASS`+profile/回退开关；落地前补各自 detailed plan。刻意不写 bite-sized 代码——依赖的框架 API 尚不存在，硬写会成占位符。）

---

## Self-Review

**Spec coverage**（对照 spec §8 修订）:
- §8.1 事实订正（`num_experts_per_rank` / quant_mode 固定 fp8 / TDM contiguous 默认）→ Global Constraints + Task 1 ✓
- §8.3 R1（contiguous、保留 topids/psum/ep_rowmap）→ Task 1–4（1a）✓
- §8.2 前缀和/顺序张力 → 1a 用到达序 per-token 规避；grouped 落位归入 1b ✓
- 1b（消除 launch）→ 里程碑，待 1a 后补 plan ✓
- spec §4/§7（Phase 2/3）→ 里程碑，各补独立 plan ✓

**Placeholder scan:** Task 2/3 的 device kernel body 标注"复刻 `_get_compiled_fused_quant_preshuffle` 的 quant 数学 / `_grouped_a8w4_preshuffle_e8m0_scale` 的 preshuffle 索引"——给出确切参考函数名，执行者按名定位复刻，非可省略 TBD；harness comm bootstrap 从 `test_mega_moe.py` 抽（真实 API）。

**Type consistency:** `disp_out_q_view() -> (payload, scale)`（Task 1 定义 / Task 2 断言 / Task 3 消费）；`fuse_dispatch_gemm1` / `_fused_q_regions` / `in_fp8_payload`+`in_fp8_scale` 全程同名。到达序 slot 统一用 `dest_tok_id`。
