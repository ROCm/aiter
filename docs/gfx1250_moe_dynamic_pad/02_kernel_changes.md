# 02 · kernel 改动(gemm_mxscale_gfx1250.py)

status: not-started
owner: (认领后署名)
依赖:`01`(需要 `make_tensor_descriptor_2d` 的新 `valid_inner/valid_outer` API + OOB 语义结论)

目标文件:`/root/00_code/aiter/aiter/ops/flydsl/kernels/gemm_mxscale_gfx1250.py`
(kernel builder `compile_mxscale_gemm`,~3261 行)

> ⚠️ 先看 `05`:确认生产 grouped MoE 是走这个文件,还是 `moe_gemm_2stage_mxscale_gfx1250.py`。
> 若是后者,本任务的 §改动点要平移到那个文件(结构类似:都用 TDM 2D + 描述符 make_desc_*)。

---

## 设计要点(基于 OOB,不动计算)

核心思想:**不照搬 MI350 的静态 skip**。用 TDM 2D OOB:
- K-pad → load 端把 K(inner)有效 bound 设为 remaining → 末尾自动补零 → MFMA 算到 0。
  **`num_k_tiles` / `loop_iters` / `make_tail_plan` / `compute_tile` / pipeline 全部不改。**
- N-pad → store 端把列有效 bound 设为 `N_valid` → 硬件丢弃越界列 → **保留 TDM-store 快路径**。

---

## 改动点清单

### A. 新增 runtime 标量(kernel 签名)
当前 entry(`:471-486`)只有 `i32_m` / `i32_n`:
```python
def kernel_mxscale_gemm(..., arg_m_tile_prefix, arg_m_tile_map, i32_m, i32_n):
```
- 加 `i32_k_valid`(收缩维有效长度)和 `i32_n_valid`(输出列有效长度)。
- 在各 host launcher(见 §D)把它们透传。
- 先例:masked 路径已有“运行时读 `valid_m` 当谓词”(`:528-537`),加 runtime 标量是成熟模式。

> 命名按 stage 语义:stage1 `k_valid = model_dim - model_dim_pad`,`n_valid = (gate+up 的有效列)`;
> stage2 `k_valid = inter_dim - inter_dim_pad`,`n_valid = model_dim - model_dim_pad`。
> kernel 内只认 `i32_k_valid`/`i32_n_valid` 这两个抽象量,stage 差异在 wrapper 里换算(见 `03`)。

### B. K-pad:load 描述符传 remaining(inner 维)
改 `make_desc_a` / `make_desc_b` / `make_desc_as` / `make_desc_bs`(`:583-665`):
- 给每个 load descriptor 的 **inner(K)维** 传 `valid_inner = max(0, K_valid_scaled - k_base_scaled - warp_off_inner)`。
  - 注意各 tensor 的 K 单位不同:A 是 `K_packed_a`、B 是 `K_packed_b`、scale 是 `K_scale`(`/SCALE_BLOCK`)。
    remaining 要按各自的打包/缩放换算(`PACK_FACTOR_A/B`、`SCALE_BLOCK`)。
- `k_base` 在主循环里是按 `adv_*` 常量推进的(`:1968-1973`):
  `adv_a = tile_k//PACK_FACTOR_A`、`adv_b = packed_tile_k_b*16`、`adv_as/bs = tile_k//SCALE_BLOCK * rep`。
  remaining 需要随这个 k_base 走;最简单是每个 tile 都传 `valid - k_base`,前面的 tile remaining≥tile 自然不触发 OOB,只有最后一个 tile 真正裁剪。
- **scale 越界其实可不裁**:只要数据(A/B)越界补零,fragment 就是 0,`0 * 2^scale = 0`,scale 是啥都不影响。
  ⇒ 第一版可只给 **A、B 数据描述符** 传 remaining,scale 描述符先不动,降低复杂度(在 `05` 记录该简化)。
- **不需要**改 `num_k_tiles`/`loop_iters`/tail_plan/`compute_tile` 的任何循环结构。

> 前提:`01` microbench 确认 **load OOB = zero-fill**。若是“保留旧值”,改为:
> 在 prologue 对 LDS 做一次清零,或对最后一个 tile 退回静态 skip(参考 `04` 的 MI350 做法)。

### C. N-pad:store 描述符传 N_valid(列维)
TDM-store 输出描述符在 `:1887-1903`,当前:
```python
tensor_shape=(batch_count * M, N), strides=(N, 1), tile_shape=(warp_tile_m, warp_tile_n)
```
(`tensor_shape` 当前无效。)
- 给输出 store descriptor 的 **列维** 传 `valid_outer/inner = max(0, N_valid - blk_n - warp_n_off)`
  (按 descriptor 的 dim0/dim1 约定对应到列;注意 N 是 inner 维,row stride=N)。
- 这样硬件丢弃越界列 ⇒ **保留 `use_tdm_store` 快路径**,不需要逐列谓词。
- **解除原限制**:之前认为“N-pad 必须回退 scalar store”,有了 store OOB 后不需要了。
- `needs_grouped_row_masked_store`(`:449`)的逻辑保持:它处理的是 **行(M)** 掩码,与 N-pad 正交。
  N-pad **不** 需要并入它强制 scalar store。
- B load 侧的 N(行)维也可传 `N_valid` remaining(越界补零),但**可省**:越界列反正不 store。
- bias-add(`_load_bias_vec8`,`:1405-1423`)读越界列无害,因为对应输出列被 store OOB 丢弃。

> 前提:`01` microbench 确认 **store OOB = drop**。

### D. grid(纯性能,正确性已由 OOB 兜住)
- 持久化 grouped 路径 `launch_mxscale_gemm_masked_persistent`:`gx = ceil(idx_n/tile_n)`(`:2995`)。
- flat masked 路径 `launch_mxscale_gemm_masked`:`gy = ceil(idx_n/tile_n)`(`:2936`)。
- 改成用 `n_valid` 计算,少 launch 纯 padding 的 N-tile。
- kernel 内 `n_tiles_per_batch`(`:2727`)/`n_tile_in_range`(`:2759`)同步用 n_valid。
- 这步是**纯性能优化**,可放第二批;不做也正确(只是空转几个 tile,OOB 丢弃)。

### E. 多 warp 注意
grouped 默认 `m_warp=1, n_warp=4`(N 跨 4 warp)。每个 warp 的 remaining-N 不同:
- `make_tensor_descriptor_2d` 内部已经算了 `warp_off_outer/inner`(`:289-295`),
  但 remaining 的减法需要在“传入 bound”时就考虑 per-warp 偏移,或者让 flydsl 在内部用 warp_off 修正。
- 与 `01` 的 owner 对齐:**remaining 减 warp_off 这步放 flydsl 内部做** 更干净
  (调用方只传全局 `K_valid - k_base` / `N_valid - n_base`,flydsl 再减 warp_off)。建议这样定 API。

---

## 验收
- a8w4 / mxfp4 两个 format 的 grouped stage1+stage2,带 `model_dim_pad>0` 和 `inter_dim_pad>0`,数值正确。
- 不退化无 pad 的现有性能(pad=0 时 remaining≥tile,OOB 不触发,路径等价)。
- 与 `03`、`05` 对齐联调。
