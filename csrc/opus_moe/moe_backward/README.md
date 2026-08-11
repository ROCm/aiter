<!--
Opus MoE Backward — 实现计划 / build checklist
本目录隔离 backward MoE 的 C++/HIP 代码,不混进 gfx950/a8w4/ 前向。
-->

# Opus MoE Backward 实现计划（aiter / gfx950）

## 0. 目标与范围

**目标**：在 aiter 里用 opus 写单卡 gfx950 的 MoE **反向**算子，配套现有 a8w4
前向。默认 **BF16 计算 + FP32 梯度累加**（K3 路线），产出 `dx / dW1 / dW2 / dp`。

**本期不做**（留后续）：EP/all-to-all 反向、FP8 反向 + transpose-requant、
prefill 深度调优。

**关键既定事实**（探索确认）：
- opus MoE 前向**只有 stage2（down-proj）是 live**；stage1 的 `.py` 已删（残留仅
  `__pycache__`/build）。故 stage2 a8w4 decode 栈是唯一可照抄的前向模板；stage1
  反向要从零起。
- 绑定 = **pybind11**（`@compile_ops(ffi_type="pybind")`）+ 自研 JIT
  （`aiter/jit/core.py` + `optCompilerConfig.json`）+ 预编译 codegen
  （`csrc/opus_moe/gen_instances.py` 读 `moe_stage2_a8w4_meta.py` 生成 C++ manifest）。
- backward 惯例（rope/mha 证实）：kernel 与 fwd **同层放置、文件名区分**、
  **pybind/JIT module 独立**（`module_<name>_bwd`）。本期为了隔离，C++ 放到
  **本目录 `moe_backward/`**（cross-dir include 复用前向 utils），module 用
  `module_moe_opus_bwd`。

---

## 1. 反向骨架（一句话回顾，详见 backward walkthrough）

一个 MoE 反向 = **4 个 expert GEMM + 激活反向 + combine 反向 + router 反向**：

| 算子 | 数学 | 类型 | 复用/参照 |
|---|---|---|---|
| Stage2 dgrad | `dh = W2ᵀ·dy` | M-grouped | 前向 stage2 结构 + Wᵀ 转置访问 |
| Stage2 wgrad | `dW2 = Σ dy·hᵀ` | **K-grouped**（新） | `ptgmm`/DeepGEMM k-grouped/SonicMoE dW2 |
| Stage1 dgrad | `dx = W1ᵀ·[dg;du]` | M-grouped | 前向 stage1（已删，重建）+ 跨 topk 归约 |
| Stage1 wgrad | `dW1 = Σ [dg;du]·xᵀ` | **K-grouped**（新） | 同 wgrad |
| 激活反向 | `dh→[dg;du]`（SiTUv2 Jacobian） | elementwise | 融进 stage2 dgrad epilogue（SonicMoE dH） |
| combine 反向 | `dy=p·dout`（广播）；`dp=<dout,y>`（colvec） | epilogue 归约 | 融进 stage2 dgrad，不物化 y |
| router 反向 | `dp→dlogits` | 小 kernel | SonicMoE topk/softmax bwd，或 opus 外 |

**dtype**：wgrad = BF16×BF16→FP32（无 scale word）；dgrad = fp4 权重 scaled-MFMA
（复用前向 E8M0 机制）但需 Wᵀ；`dW1/dW2` 输出 **FP32**（喂优化器，非 fp4）。

---

## 2. 目录与文件布局

C++/HIP 全部隔离在本目录（`csrc/opus_moe/moe_backward/`）：

```
csrc/opus_moe/moe_backward/
  README.md                              ← 本文
  opus_moe_bwd.cu                        ← 编译 TU（#include host_impl）
  gen_instances_bwd.py                   ← 反向 manifest codegen（可 M1 后再上，先手写）
  opus_moe_bwd_common.py                 ← 反向 meta 桥（读 Python meta 表）
  include/
    opus_moe_bwd.h                       ← host 函数声明（torch_itfs 命名空间）
    opus_moe_bwd_host_impl.cuh           ← host 校验/选 kid/打包 kargs/launch
    opus_moe_bwd_common.cuh              ← 反向 kargs POD 结构（dgrad/wgrad 各一）
    gfx950/a8w4/
      # ---- stage2 backward ----
      opus_moe_traits_stage2_a8w4_dgrad_gfx950.cuh
      opus_moe_stage2_a8w4_dgrad_dispatch_gfx950.cuh
      opus_moe_pipeline_stage2_a8w4_dgrad_policy_gfx950.cuh
      opus_moe_pipeline_stage2_a8w4_dgrad_main_gfx950.cuh   ← __global__ (dgrad+激活反向+dp)
      opus_moe_traits_stage2_a8w4_wgrad_gfx950.cuh
      opus_moe_stage2_a8w4_wgrad_dispatch_gfx950.cuh
      opus_moe_pipeline_stage2_a8w4_wgrad_policy_gfx950.cuh
      opus_moe_pipeline_stage2_a8w4_wgrad_main_gfx950.cuh   ← __global__ (K-grouped)
      # ---- stage1 backward（M3 起） ----
      opus_moe_..._stage1_a8w4_{dgrad,wgrad}_..._gfx950.cuh
    gfx950/opus_moe_bwd_arch_gfx950.cuh  ← launchers（*_launch_gfx950）
```

**复用前向共享头**（相对 include，跨目录）：
```
../include/gfx950/opus_moe_stage2_utils_gfx950.cuh
../include/gfx950/opus_moe_arch_gfx950.cuh
../include/opus_moe_arch.cuh
../../include/opus/opus.hpp
```

Python / 绑定 / 构建（在各自惯例位置，不进本目录）：
```
aiter/ops/opus/moe_bwd.py           ← @compile_ops 桩 + 公开 wrapper + class OpusMoEBwd*
aiter/ops/opus/moe_bwd_meta.py      ← 反向 kid 表（单一真相源，喂 codegen）
aiter/ops/opus/moe_backward_autograd.py（或并入 fused_moe.py）← OpusMoEFunc(autograd.Function)
csrc/pybind/opus_moe_bwd_pybind.cu  ← 4 行，展开宏
csrc/include/rocm_ops.hpp           ← 新增 OPUS_MOE_BWD_PYBIND 宏
aiter/jit/optCompilerConfig.json    ← 新增 "module_moe_opus_bwd"
aiter/__init__.py                   ← star-import moe_bwd，暴露 aiter.opus_moe_*_bwd
```

---

## 3. 逐层 checklist（新增/改什么、参照谁）

| 层 | 文件 | 做什么 | 参照 |
|---|---|---|---|
| Python 桩 | `aiter/ops/opus/moe_bwd.py` | `@compile_ops("module_moe_opus_bwd", fc_name="opus_moe_stage2_a8w4_wgrad", ffi_type="pybind", gen_fake=...)` 空桩 + 公开 wrapper（shape/kid 选择） | `moe_stage2_a8w4.py`、`mha.py:mha_bwd` |
| meta | `aiter/ops/opus/moe_bwd_meta.py` | 反向 kid 表（每 kid：tile BM/BN/BK、K-loop 步长、dtype、reduction 方案） | `moe_stage2_a8w4_meta.py` |
| autograd | `moe_backward_autograd.py` | `class OpusMoEFunc(torch.autograd.Function)`：forward 存 `x/pre-act(g,u)/h/p + sorting metadata`；backward 调各 `*_bwd` 返回逐输入梯度 | `mha.py:FlashAttnFunc(2478)`、`rope.py:RoPE(1119)` |
| JIT | `optCompilerConfig.json` | 加 `"module_moe_opus_bwd"`：srcs=[pybind.cu, opus_moe_bwd.cu]，extra_include 含 `opus_moe/include`、`opus_moe/moe_backward/include`、`opus_gemm/include`；`blob_gen_cmd` 指向 `gen_instances_bwd.py`（可空，先手写 manifest） | forward `module_moe_opus`（563–582） |
| pybind | `csrc/pybind/opus_moe_bwd_pybind.cu` + `rocm_ops.hpp` 宏 `OPUS_MOE_BWD_PYBIND` | `m.def("opus_moe_stage2_a8w4_wgrad", &aiter::torch_itfs::...)` 等 | `OPUS_MOE_PYBIND(rocm_ops.hpp:321)`、`MHA_BWD_PYBIND(994)` |
| C++ 声明 | `moe_backward/include/opus_moe_bwd.h` | host 函数原型（`namespace aiter::torch_itfs`） | `opus_moe/include/opus_moe.h` |
| host | `opus_moe_bwd.cu` → `include/opus_moe_bwd_host_impl.cuh` | 校验 dtype/shape、选 kid、算 grid、打包 kargs、launch | `opus_moe_host_impl.cuh` |
| kargs | `include/opus_moe_bwd_common.cuh` | dgrad/wgrad 各一个 POD（指针 + stride + expert offset/count + 维度） | `opus_moe_common.cuh` |
| device | `include/gfx950/a8w4/*_gfx950.cuh` + `gfx950/opus_moe_bwd_arch_gfx950.cuh` | 每个 GEMM 一套 traits/dispatch/policy/main + launcher | 前向 stage2 的四件套 |

---

## 4. 分阶段落地路线（里程碑）

**M0 — 金标与基线（已具备）**
- `aiter.ops.triton.moe_bwd_ref.torch_moe` 提供完整 golden
  （router+dgrad+激活反传+wgrad），作为 kernel 的正确性 oracle。
- 验收：silu/gelu/swiglu/situv2 四激活、stage1/stage2、`dW/dA err=0`。

**M1 — opus Stage2 wgrad（第一个 opus kernel，最高价值/风险）**
- 只做 `dW2 = Σ dy·hᵀ`，K-grouped，BF16×BF16→FP32，单 CTA 全 K 归约（无 atomic）。
- 交付：meta kid 表 1 条、host、kargs、traits/dispatch/policy/main、pybind、JIT
  module、Python 桩+wrapper、test tier。
- 验收：`opus dW2` 对 `dW2_ref` err=0；roofline 记录（关注 K=route 利用率，训练 T）。
- 详细步骤见 §5。

**M2 — opus Stage2 dgrad + 激活反向 + dp（融合 dH kernel）**
- `dh = W2ᵀ·dy`（复用前向 stage2 route-block/scaled-MFMA + Wᵀ 转置访问），epilogue
  融 SiTUv2 Jacobian 出 `[dg;du]`、colvec 出 `dp`、广播 `dy=p·dout`——一趟不物化 y/dy。
- 验收：`dh/[dg;du]` 对 golden、`dp` 对 golden；err=0。

**M3 — opus Stage1 dgrad + wgrad**
- `dx = W1ᵀ·[dg;du]`（+ 跨 topk 归约回 token）、`dW1 = Σ [dg;du]·xᵀ`（K-grouped，
  x gather-fusion）。stage1 前向已删 → pre-act `g,u` 来源：先走"前向额外存 g,u（bf16）"
  路线（改前向 stage1 或在 autograd.forward 里补存）。
- 验收：`dx/dW1` 对 golden err=0；dW1 gate/up 布局对齐前向 W1（interleave/concat）。

**M4 — 端到端 autograd 打通 + router 反向**
- `OpusMoEFunc` 串起 forward（存中间量）→ backward（4 GEMM + 激活 + combine）；
  `dp→dlogits` router 反向（小 kernel 或 opus 外）。
- 验收：整层 `torch.autograd.grad` 对 torch 参考 err=0；接入 fused_moe 训练路径。

**M5 — 调优与扩展**
- kid 表扩容（多 tile 配置）、gen_instances_bwd codegen、wgrad split-K（K 大时）、
  roofline 达标（对齐/超过 triton `ptgmm`、Primus）；可选 fp8 反向 + transpose-requant。

依赖链：M0 → M1 → M2 → M3 → M4 → M5（M1/M2 可小幅并行，但都依赖 M0 金标）。

---

## 5. M1 详细步骤（Stage2 wgrad，文件级）

**数学**：`dW2[e] = Σ_{route∈e} dy_j · h_jᵀ`，输出 `[D,H]=[3584,384]` FP32。
grid = `(expert, D-tile, H-tile)`；CTA 内 K-loop 遍历该 expert 的 `count` 条 route。

1. **meta**（`moe_bwd_meta.py`）：加 `OPUS_A8W4_STAGE2_WGRAD_BY_KID`，字段
   `{BM(沿D), BN(沿H), BK(K-loop 步长), num_warps, reduction:"single_cta"}`；helper
   `opus_a8w4_stage2_wgrad_kid(shape)`。
2. **kargs**（`opus_moe_bwd_common.cuh`）：
   `{ const bf16* dy; const bf16* h; float* dW2; const int* sorted_token_ids;
      const int* sorted_expert_ids; const int* expert_frequency_offset;
      int D, H, num_experts; int64_t stride_dw2_e, stride_dy_*, stride_h_*; }`
3. **host**（`opus_moe_bwd_host_impl.cuh`）：校验 dy/h 为 bf16、dW2 为 fp32；
   grid = `(E, cdiv(D,BM), cdiv(H,BN))`；选 kid；打包 kargs；调 launcher。
4. **traits**（`..._wgrad_gfx950.cuh`）：M=BM(D 行 tile)、N=BN(H 列 tile)、
   K=route；MFMA 16x16x16(BF16)；无 scale word/selector。
5. **dispatch**：`switch(kid)` 实例化 traits 调 launcher（先手写 1 个 kid，
   codegen 后置）。
6. **main**（`__global__`）：
   ```
   读 blockIdx → (expert e, D-tile, H-tile)
   start=expert_frequency_offset[e]; count=offset[e+1]-start
   acc[BM,BN] = 0 (FP32)
   for r in [start, start+count):            # K-loop = route
       row = sorted_token_ids[r]; 解码 (token,slot); skip padding
       load dy_j[D-tile]  (bf16, 从紧凑源 gather)
       load h_j[H-tile]   (bf16; a2 反量化或前向存的 bf16 h)
       acc += mma(dy_jᵀ, h_j)                # BF16→FP32
   写 dW2[e, D-tile, H-tile] = acc           # 单 CTA 全 K，无 atomic
   ```
7. **launcher**（`opus_moe_bwd_arch_gfx950.cuh`）：算 grid/block，launch kernel。
8. **pybind**：`opus_moe_bwd_pybind.cu` 展开 `OPUS_MOE_BWD_PYBIND`；宏在
   `rocm_ops.hpp` 加 `m.def("opus_moe_stage2_a8w4_wgrad", ...)`。
9. **JIT**：`optCompilerConfig.json` 加 `module_moe_opus_bwd`。
10. **Python**：`moe_bwd.py` 桩 `_opus_moe_stage2_a8w4_wgrad_raw` + wrapper
    `opus_moe_stage2_a8w4_wgrad(dy, h, sorted_*, ...) -> dW2`；`aiter/__init__.py` 导出。
11. **验证**：对 `torch_moe` autograd 金标检查完整梯度，性能使用 HIP event 计时。

---

## 6. 验证策略

- **正确性**：`torch_moe` autograd 金标 + `checkAllclose err=0`；覆盖 Opus/Triton、
  方阵/非方阵、expert/full router 梯度。
- **性能**：roofline（TFLOPS/GB/s/%peak，peak≈1300），对比 naive floor / triton
  `gmm`/`ptgmm` /（可选 Primus）；wgrad 重点看 K=route 利用率（训练 T，非 decode）。
- **容器**：`yifehuan_moe_backward`（`docker start` 先；**注意会被外部删，可能需按
  记忆里的命令重建**，且别装 Primus 覆盖 rocm torch）。
- **回归**：前向零改动（module 独立）；`test_moe_2stage.py` 应不受影响。

---

## 7. 风险与取舍

1. **wgrad K-grouped 是全新形态**（grid/CTA/reduction 与前向都不同）→ 最高风险，
   **先做（M1）并先对齐金标**；先用单 CTA 全 K（简单正确），K 大再 split-K。
2. **dgrad 需 Wᵀ 转置访问 fp4 packed**（前向没有）→ 效率风险；M2 先"能跑对"，
   必要时先用 bf16 权重副本 / 生成转置副本，后续再优化 packed 转置访存。
3. **stage1 前向已删** → stage1 反向的 `g,u` 无处取；M3 先走"前向额外存 pre-act
   (bf16)"，别一上来就重算。
4. **gen_instances codegen 复杂** → M1 先手写单 kid manifest，codegen（gen_instances_bwd.py）
   放到 M5，避免早期被 codegen 挡住。
5. **容器易失 + Primus 会污染 torch** → 复现步骤严格按记忆
   [[moe-backward-harness]]；实验产物写 repo 外 data/。
6. **fp8 反向 + transpose-requant** → 明确留到 M5，别和 BF16 主线纠缠。

---

## 8. 关联索引

- 前向模板文件：`csrc/opus_moe/include/gfx950/a8w4/*`、`opus_moe_host_impl.cuh`、
  `opus_moe_common.cuh`、`gen_instances.py`、`aiter/ops/opus/moe_stage2_a8w4*.py`
- backward 惯例模板：rope（`csrc/kernels/rope/*_bwd_kernels.cu` + `rope.py`）、
  mha（`csrc/py_itfs_ck/mha_bwd_kernels.cu` + `mha.py:FlashAttnFunc`）
- grouped GEMM 原语：`aiter/ops/triton/gmm.py`（`gmm`=M-grouped dgrad、`ptgmm`=K-grouped wgrad）
- 外部参照：SonicMoE（`sonic-moe/sonicmoe/functional/backward.py` 融合 dH kernel）、
  DeepGEMM（`k_grouped_fp8_gemm_tn_contiguous`）

---

## 9. Sonic 语义完整反向与公平性能对比

`opus_moe_ref` / `triton_moe_ref` 以预计算的 router logits 为输入，适合测量
expert + routing-score backward。`opus_moe` / `triton_moe` 进一步在图内执行
`router_logits = F.linear(x, router_w)`，因此还会生成 `drouter_w`、router 分支的
`dx`，并与 expert `dx` 合并。

公平对比路径使用：

- `SonicSwiglu`：标准 `silu(gate) * up`，不改变原有 GPT-OSS 风格 `Swiglu`；
- FP32 top-k routing scores；
- BF16 dW/dx 直接输出，不做 BF16→FP32→BF16 往返；
- Triton GMM 显式 transposition，支持 `H == 2I` 方阵；
- HIP event、固定 warmup 和显式 repeat 数。

性能报告区分 `expert`（终止于 dlogits）与 `full`（包含 router projection
backward）两个范围；有效 TFLOP/s 使用 SonicMoE 的 expert backward 分母
`12*T*H*I*K`。
