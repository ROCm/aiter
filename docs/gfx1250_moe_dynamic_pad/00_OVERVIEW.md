# gfx1250 (MI450) MoE 动态 pad 支持 — 总览

> 目标:在 gfx1250 的 grouped MoE GEMM 上支持 `model_dim_pad` / `inter_dim_pad`，
> 用 **运行时(动态)参数 + TDM 2D 硬件 OOB**，而不是照搬 MI350(CDNA)的静态编译期 skip。

本目录是给多个 agent 并行协作用的任务/上下文记录。每个 agent 认领一个子任务文件，
先读 `00_OVERVIEW.md` 再读自己那份。

---

## TL;DR 关键结论(必读)

1. **MI350 的做法是静态的**:`mixed_moe_gemm_2stage.py` 把 padded dim 作为编译期常量，
   用 `_pad_ku_skip`/`_tail_ku`/`ku_count` 在最后一个 K-tile 里跳过 padding 子步;N 方向靠 host 少 launch tile。
   每个 pad 形状都要重新编译(`@functools.lru_cache`)。

2. **gfx1250 硬件 TDM 2D descriptor 支持 OOB,且支持 runtime(SSA)动态值**——
   但 **flydsl 当前的 `make_tensor_descriptor_2d` 没把这能力接出来**:
   它把 `tensor_shape` 入参解包后丢弃,把 descriptor 的 `tensor_dim0/dim1` 直接设成 per-warp tile 大小
   (`tensor_dim == tile_dim` ⇒ 永不越界)。gather 路径 (`make_tensor_gather_descriptor`) 则真正用了
   `tensor_dim1` 做 OOB,并支持 runtime SSA(`_td1_is_runtime`),证明硬件这条路可行。

3. **因此动态方案的形态**:
   - 先给 flydsl 的 2D descriptor **接通一个(可 runtime 的)有效 tensor bound** → 写入 `tdim`。【阻塞项】
   - kernel 侧加两个 runtime 标量 `i32_k_valid` / `i32_n_valid`,各 descriptor 传 “剩余有效长度”。
   - **K-pad 走 load OOB zero-fill** → MFMA 算到 0,**compute/pipeline/tail_plan 完全不动**。
   - **N-pad 走 store OOB drop** → 保留 TDM-store 快路径,**不需要逐列谓词、不回退 scalar store**。
   - grid 用 `N_valid` 计算,少 launch 纯 padding tile(纯性能)。

4. **唯一硬性风险**:TDM 2D **load 越界到底是补 0 还是保留 LDS 旧值**、以及 OOB 的**参考系**
   (相对 descriptor 折叠后的基址,而非全局 origin)。必须用 microbench 实测确认。见 `01`。

---

## Stage / pad 角色映射(两个 stage 互换,务必记牢)

| | 收缩维 K | 输出维 N |
|---|---|---|
| **stage1** (gate+up) | `model_dim`(pad = `model_dim_pad`) | `inter_dim`(×2 gate+up,pad = `inter_dim_pad`) |
| **stage2** (down)    | `inter_dim`(pad = `inter_dim_pad`) | `model_dim`(pad = `model_dim_pad`) |

“dims include pad” 语义:传进来的 `model_dim`/`inter_dim` 已含 padding;`valid = dim - pad`。
Tensor / scale 形状按 padded dim 分配(已经是这样,不改)。

---

## 文件 / 模块地图

| 文件 | 角色 |
|---|---|
| `aiter/ops/flydsl/kernels/gemm_mxscale_gfx1250.py` | **kernel builder** `compile_mxscale_gemm`(fp4/fp8/a8w4),grouped MoE 也走它 |
| `aiter/ops/flydsl/kernels/moe_grouped_gemm_mxscale_gfx1250.py` | grouped MoE stage1/stage2 **wrapper**,构 m_tile_map/prefix |
| `aiter/ops/flydsl/grouped_moe_gfx1250.py` | **top-level 入口** `_maybe_grouped_gfx1250_a8w4_moe`,pad 当前被清零 |
| `aiter/ops/flydsl/kernels/moe_gemm_2stage_mxscale_gfx1250.py` | ⚠️ 另一个 gfx1250 MoE 实现(用 gather + runtime token dim)。**需先确认生产路径用的是哪一个**,见 `05` |
| `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` | **MI350/CDNA 参考实现**(静态 pad),见 `04` |
| `FlyDSL/build-fly/python_packages/flydsl/expr/rocdl/tdm_ops.py` | **flydsl TDM op**(`make_tensor_descriptor_2d` / `make_tensor_gather_descriptor`) |

> 注:flydsl 是从 `/root/00_code/FlyDSL/build-fly/python_packages/flydsl` symlink 过去的(可直接改源码)。

---

## 任务拆分与依赖

```
01_flydsl_tdm_oob   ← 阻塞项 / 关键路径(先做,带 microbench 验证)
        │
        ├──> 02_kernel_changes        (依赖 01 的 API 与 OOB 语义确认)
        │
        └──> 03_wrapper_and_entry     (可与 02 并行起草,落地依赖 02 的新标量签名)

04_reference_mi350   (只读参考,随时可看,无依赖)
05_risks_and_validation  (贯穿全程;01 的 microbench 结论回写这里)
```

建议顺序:**先把 `01` 的 microbench 结论敲定**(决定 K-pad 能否走 zero-fill 捷径),
再并行推进 `02` / `03`。

---

## 状态板(各 agent 更新)

- [ ] 01 flydsl: 2D descriptor 接通 runtime tensor bound + OOB microbench 验证
- [ ] 02 kernel: 新增 runtime 标量 + 描述符传 remaining + K/N pad + grid
- [ ] 03 wrapper/entry: 停止清零 pad,透传 valid dims 到 launch
- [ ] 05 验证: 数值对拍(对照 CDNA / 无 pad 基线)+ 性能

---

## agent 协作约定

- 改代码前先在对应 md 顶部把 `status` 改成 `in-progress` 并署名。
- 关键发现(尤其 `01` 的 OOB 实测结论)回写到 `05_risks_and_validation.md`,其它人据此调整。
- 所有 file:line 引用都标了大致行号,**以当前源码为准**(文件在持续改动)。
