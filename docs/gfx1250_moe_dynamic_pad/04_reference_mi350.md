# 04 · MI350 (CDNA) 静态 pad 参考实现(只读对照)

status: reference-only

文件:`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`
作用:理解 MI350 怎么做 pad,作为正确性对照 / 语义来源。**我们不照搬这套静态方案**,
但数值语义必须一致(用于 `05` 对拍)。

---

## 语义(`:202-207`)
```
model_dim = model_dim_true + model_dim_pad   (K direction, stage1)
inter_dim = inter_dim_true + inter_dim_pad   (N direction, stage1)
```
- 传进来的就是 padded dim;tensor 大小用 padded dim;`_valid = dim - pad`。
- padding 在每个维度的**高位尾部**。

## K-pad(静态跳过最后一个 K-tile 的尾部子步)

stage1(`:863-873`):
```python
_K_per_ku   = tile_k // k_unroll
_pad_k_elems = (model_dim_pad % tile_k) if (not _is_splitk and model_dim_pad > 0) else 0
_pad_ku_skip = _pad_k_elems // _K_per_ku
_tail_ku     = k_unroll - _pad_ku_skip
```
- 在最后一个 K-tile 用 `ku_count=_tail_ku` 截断 MFMA 内层循环(`compute_tile` `:1298-1316`,消费点 `:1873`/`:1929`)。
- 只裁**最后**几个子步(padding 在高位)。split-K 下禁用(靠 B padding 为 0)。

stage2(`:3554-3562`,角色:K=inter_dim):同样逻辑,由 `inter_dim_pad` 驱动,无 split-K guard。消费点 `:4213`/`:4286`。

## N-pad(靠 host grid 少 launch,不在 kernel 逐列掩码)

- `_inter_dim_valid`(`:207`)、`check_c_n_valid_gate`/`check_c_k_valid_gate`(`:3090-3094`)都是**算了但没用**的残留。
- 真正机制在 host:
  - stage1 launcher(`:2704-2723`):`gx` 用 `inter_in - 2*inter_dim_pad + tile2_pad` 算 N-tile 数;
    `tile2_pad` 额外把有效 inter_dim 向上对齐到 `tile_k/2`(给 stage2 的 K-tile 边界对齐用)。
  - stage2 launcher(`:4549-4552`):`gx = ceil((n_in - model_dim_pad)/tile_n)`。
- 部分重叠的 tile 把 padding 列写进 padded 输出区(无害);纯 padding tile 不 launch。
- xcd_swizzle 时 kernel 内会重算 `_gx`(stage1 `:534` 减 `2*inter_dim_pad`;stage2 `:3121` 减 `model_dim_pad`)。

## 角色互换表
| | K | N |
|---|---|---|
| stage1 | model_dim(pad=model_dim_pad) | inter_dim×2(pad=inter_dim_pad) |
| stage2 | inter_dim(pad=inter_dim_pad) | model_dim(pad=model_dim_pad) |

## 为什么 gfx1250 不照搬
- 静态:每个 pad 形状要重编译(`@functools.lru_cache`)。
- 需要侵入 compute/pipeline(`ku_count` 截断)。
- gfx1250 有 TDM 2D OOB,可做成**运行时**且**几乎不碰计算**——见 `02`。

## scale-K 对齐(注意,和 dim-pad 是两回事)
`scale_k_padded = (inter_dim+255)//256*256`(`:2894`,用于 `:3101` 等)是 `e8m0_shuffle` 把 scale group 维
round 到 256 的对齐,**不是** `model_dim_pad`/`inter_dim_pad`。换算 gfx1250 的 scale 描述符时别混淆。
