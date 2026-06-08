# 03 · wrapper + 入口层 plumbing

status: in-progress
owner: claude-agent
依赖:`02`(需要 kernel 新增的 `i32_k_valid`/`i32_n_valid` 标量签名最终确定)

涉及文件:
- `aiter/ops/flydsl/grouped_moe_gfx1250.py`(top-level 入口)
- `aiter/ops/flydsl/kernels/moe_grouped_gemm_mxscale_gfx1250.py`(stage1/stage2 wrapper)

---

## A. 入口:停止清零 pad — `grouped_moe_gfx1250.py`

`_maybe_grouped_gfx1250_a8w4_moe`(~`:212`)当前在 ~`:301-304` **把 pad 直接清零并继续**:
```python
if hidden_pad != 0 or intermediate_pad != 0:
    hidden_pad = 0
    intermediate_pad = 0
    _grouped_dbg("haspad")
```
改动:
- 不再清零;把 `hidden_pad → model_dim_pad`、`intermediate_pad → inter_dim_pad` 透传下去。
- tensor / scale 形状(~`:540-628`)已按 **padded** `model_dim`/`inter_dim` 构建,**保持不动**(符合 “dims include pad” 语义)。
- 在 launch 时计算并传入 runtime 有效维(见 C)。

## B. wrapper:加参数并透传 — `moe_grouped_gemm_mxscale_gfx1250.py`

- `_compile_base_a8w4_gemm`(~`:682`)及 stage1/stage2 的 `compile_moe_grouped_gemm{1,2}_*_masked`
  (~`:725` / `:1016`)加 `model_dim_pad` / `inter_dim_pad` 参数,透传到底层 `compile_mxscale_gemm`。
- `_validate_common`(~`:64-96`)已要求 `model_dim%32==0`、`inter_dim%32==0`、`tile_k%128==0`:
  **padding 量必须保证 padded dim 仍是 32 的倍数**(MXScale block),否则 scale 对不齐。加一条 assert 明示。
- `_preshuffled_scale_shape`(~`:261-278`)按 padded dim 推 scale 形状,保持自洽,无需特殊处理。

## C. launch 时计算 runtime 有效维并传入

kernel 现在的标量是 `i32_m`/`i32_n`,本任务加 `i32_k_valid`/`i32_n_valid`(命名以 `02` 为准)。
按 stage 角色换算(见 `00` 的映射表):

| | `i32_k_valid` | `i32_n_valid` |
|---|---|---|
| stage1 | `model_dim - model_dim_pad` | gate+up 的有效列(注意 layout:`gguu`/`gugu`、是否 `2*inter_dim`) |
| stage2 | `inter_dim - inter_dim_pad` | `model_dim - model_dim_pad` |

注意:
- stage1 的 N 是 gate+up 拼接,`fused_n` 取决于 `stage1_weight_layout`(`gugu`→`2*inter_dim`,`gguu`→`inter_dim`),
  且 fused 激活 kernel 与 raw kernel 的 N 不同(见 wrapper ~`:781-807`)。换算 n_valid 时要分清楚走的是哪个 base。
- 这些是 host 端能拿到的运行时值(`max_m`、`inter_dim`、`model_dim` 已是运行时传参),
  直接在 `_run_compiled` 的 `.launch(...)` 调用处补两个标量参数即可。

## D. 与现有 launch 路径对齐

- grouped 默认走 persistent 路径(`grouped_persistent_m=True`)→ `launch_mxscale_gemm_masked_persistent`。
- 该 launcher 的签名也要加 `i32_k_valid`/`i32_n_valid`(由 `02` 改 kernel signature 后同步)。
- grid 用 `n_valid`(`02` §D)。

---

## 验收
- 入口接收非零 `hidden_pad`/`intermediate_pad` 时不再清零,完整跑通 stage1+stage2。
- 两个 stage 的 `k_valid`/`n_valid` 换算正确(用 `05` 的对拍测试验证)。
