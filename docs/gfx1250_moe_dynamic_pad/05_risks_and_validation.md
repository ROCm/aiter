# 05 · 风险、开放问题与验证

status: living-doc(贯穿全程,各 agent 回写结论)

---

## 阻塞性风险(必须先解)

### R1 · TDM 2D load 越界是否 zero-fill?  【阻塞 02 的 K-pad 捷径】
- 决定 K-pad 能否“算到 0”。由 `01` microbench (a) 确认。
- 结论:______(待 `01` 回写:zero-fill / 保留旧值)
- 若“保留旧值”:回退方案 = prologue 清零 LDS,或最后一个 tile 用静态 skip(照 `04`)。

### R2 · TDM 2D store 越界是否 drop?  【阻塞 02 的 N-pad TDM-store 路径】
- 决定能否保留 TDM-store 快路径、免逐列谓词。由 `01` microbench (b) 确认。
- 结论:______(待 `01` 回写)
- 若“不 drop / 会写脏”:N-pad 退回 scalar buffer store + 逐列 `col < n_valid` 谓词
  (并入 `needs_grouped_row_masked_store` 走 scalar 分支)。

### R3 · OOB 参考系
- bound 是相对 descriptor 折叠基址的 **remaining**(`K_valid - k_base`),还是全局 origin?
- 由 `01` microbench (c) 确认。结论:______
- 影响 `02` 传给描述符的值的算法,以及 flydsl API 是否在内部减 `warp_off`。

---

## 次级风险

### R4 · 生产路径到底是哪个 kernel 文件?  【先确认,决定 02 改哪儿】
仓库里有两个 gfx1250 MoE 实现:
- `gemm_mxscale_gfx1250.py`(`compile_mxscale_gemm`)— wrapper `moe_grouped_gemm_mxscale_gfx1250.py` import 的是它。
- `moe_gemm_2stage_mxscale_gfx1250.py`— 用 gather + runtime token dim(`tensor_dim1=_tokens_dim1`)。
- **行动**:确认当前 grouped MoE(`grouped_moe_gfx1250.py` 实际调用链)落到哪个;`02` 的改动点平移到对应文件。
- 结论:______

### R5 · scale 描述符是否需要裁 K?
- 论证:数据越界补 0 ⇒ fragment=0 ⇒ `0 * 2^scale = 0`,scale 越界无所谓。
- ⇒ 第一版只给 A/B 数据描述符传 remaining,scale 描述符不动。需在 R1 成立(load 补零)前提下才有效。
- 确认:______

### R6 · 多 warp 的 per-warp remaining(n_warp=4)
- N 跨 4 warp,各 warp remaining-N 不同。建议 flydsl API 内部用已算好的 `warp_off`(`tdm_ops.py:289-295`)
  修正,调用方只传全局 `valid - base`。与 `01`/`02` owner 对齐 API。

### R7 · padded dim 必须 %32==0
- MXScale 32-block;`_validate_common`(wrapper `:73/77`)。pad 量需保证 padded dim 仍 %32==0。

### R8 · kernel 硬约束
- `K % tile_k == 0`(`gemm_mxscale_gfx1250.py:220-221`):padded K 仍须 %tile_k==0。
- `num_k_tiles >= num_buffers`(`:260-264`):本方案不减 num_k_tiles,天然满足。

---

## 验证计划

### 数值对拍
- 基线 A:**无 pad**(pad=0)结果,确认改动对 pad=0 路径零回归。
- 基线 B:**MI350/CDNA**(`mixed_moe_gemm_2stage.py`)同 shape + 同 pad 的结果对拍。
- 覆盖:a8w4 与 mxfp4 两个 format;stage1(silu/swiglu;gguu/gugu)与 stage2;
  `model_dim_pad>0`、`inter_dim_pad>0`、两者同时;pad < tile_k 与 pad ≥ tile_k 两类。
- 容差:参考已有记忆 `moe-fp8fp4-preexisting-logits-diff`(fp8 路径本就有 ~0.0048 的 logits_diff,
  属既有容差,别误判为 pad 回归)。

### 现有测试骨架(起点)
- `op_tests/` 下搜 grouped moe / mxscale / gfx1250 相关用例。
- 仓库根 / op_tests 有 `REFACTOR_MOE_LEGACY_UT_PLAN.md` 等既有计划可参考组织方式。

### 性能
- 对比无 pad、本方案(pad>0 但 OOB)、以及(若实现)grid 缩减后的吞吐。
- 关注:OOB 是否带来额外 TDM 开销;最后一个 K-tile 仍 DMA 全 tile(只是补零),确认可接受。

---

## 结论回写区(各 agent 更新)
- R1 load OOB: ______
- R2 store OOB: ______
- R3 参考系: ______
- R4 生产 kernel 文件: ______
- R5 scale 简化是否成立: ______
