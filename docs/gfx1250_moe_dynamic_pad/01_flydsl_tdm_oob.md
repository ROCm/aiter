# 01 · flydsl TDM 2D 接通 runtime OOB(关键路径 / 阻塞项)

status: done
owner: claude-agent

> 这是整个动态 pad 方案的地基。在本任务确认 OOB 语义之前,`02`/`03` 的设计都是假设性的。

目标文件:`/root/00_code/FlyDSL/build-fly/python_packages/flydsl/expr/rocdl/tdm_ops.py`

---

## 1. 现状:2D 路径根本没启用 OOB

`make_tensor_descriptor_2d`(`tdm_ops.py:201`):

- 第 269 行 `outer_size, inner_size = tensor_shape` 解包后 **`tensor_shape` 再没被用过**(纯文档参数)。
- 第 341–344 行把 descriptor 的张量维直接设成 per-warp tile 大小:
  ```python
  tdim0 = bpw_inner   # innermost extent per warp
  tdim1 = bpw_outer   # outermost extent per warp
  tile_d0 = bpw_inner
  tile_d1 = bpw_outer
  ```
- 第 389–402 行把 `tdim0/tdim1` 编码进 g1_s1/s2/s3(位域见注释)。
- 因为 `tensor_dim == tile_dim`,硬件认为“张量恰好等于这一块 tile” ⇒ **永远不越界**,padding 区按原样读/写。

global offset(第 303–308 行)被**折进了原始地址** `glb_addr_i64`,descriptor 没有单独的“坐标 vs origin”字段。

## 2. 证据:硬件支持 OOB + runtime,gather 路径已用

`make_tensor_gather_descriptor`(`tdm_ops.py:422`):

- docstring 引用 ISA §4.10.3.2:「row index ≥ `tensor_dim1` 视为 OOB」。
- `tensor_dim1` **支持 runtime SSA**:`_td1_is_runtime = not isinstance(tensor_dim1, int)`(约 `:531`),
  用 `arith.ori/shli/shrui` 把 SSA 值打进描述符位域。
- 生产 MoE(`moe_gemm_2stage_mxscale_gfx1250.py`)已用它对 token 维做 runtime OOB
  (`tensor_dim1=_tokens_dim1`,源自 kernel 标量 `i32_tokens_in` + `readfirstlane`)。

⇒ 硬件 descriptor 格式有 tensor 维字段、能 OOB、能 runtime;2D helper 只是没接。

## 3. 要做的改动

给 `make_tensor_descriptor_2d` 增加一个**可选的、可 runtime 的有效 tensor bound**,
覆盖默认的 `tdim0/tdim1 = bpw_*`,并按 gather 的 `_td1_is_runtime` 模式支持 SSA i32:

- 新增入参(命名待定),建议:`valid_inner=None`、`valid_outer=None`
  (None 时退化为当前行为 `= bpw_*`,保证向后兼容)。
- 把 g1_s1/s2/s3 的 `tdim0/tdim1` 编码从“纯 Python 常量”改成“常量 **或** SSA”两条路
  (参考 `workgroup_mask` 在 `:381-387` 已有的 int/ir.Value 双分支写法,以及 gather 的 SSA 位域拼装)。

### ⚠️ 参考系(最容易踩坑)
因为 offset 折进了基址(`:303-308`),descriptor 的“坐标 0”= 本 tile 的左上角。
所以传进来的有效 bound 必须是 **“从本 descriptor 基址起还剩多少有效元素”**:
```
valid_inner(本tile) = max(0, K_valid - 当前 k_base - warp_off_inner)
valid_outer(本tile) = max(0, N_valid - 当前 n_base - warp_off_outer)
```
不是全局 `K_valid`/`N_valid`。这点必须在 microbench 里验证清楚(见下)。

> 备选实现:如果硬件 OOB 必须相对全局 origin,则 2D helper 需要改成
> “不折叠 offset、单独编码坐标 + 全局 tensor_dim”。先用 microbench 判定走哪条。

## 4. microbench(决定方案成立与否,必须先做)

写一个最小 flydsl kernel(或复用现有 TDM 测试骨架),逐一确认:

- **(a) load OOB ⇒ zero-fill?** 让 tile 的 inner 维超出 valid bound,检查 LDS 越界区是 0 还是旧值。
  - 若**补 0**:K-pad 走 zero-fill 捷径成立(`02` 的核心假设)。
  - 若**保留旧值**:K-pad 需改为“先清零 LDS”或回退 skip 方案 → 回写 `05`,通知 `02`。
- **(b) store OOB ⇒ drop?** 让 store tile 的列超出 valid bound,确认越界列**没有写**到 global。
  - 成立则 N-pad 保留 TDM-store 快路径(`02` 的核心假设)。
- **(c) 参考系**:验证 bound 是“相对本 descriptor 基址的 remaining”,而非全局 origin(见 §3 警告)。
- **(d) runtime SSA**:bound 用 kernel 标量(i32)传入,确认编译/运行正确(对照 gather 已有用法)。
- **(e) 多 warp**:`num_warps>1` 时每个 warp 的 remaining 不同,确认 per-warp bound 编码正确。

参考测试位置:仓库里搜 `test_*tdm*` / `tensor_load_2d` / `tensor_store_2d` 找现有骨架。

## 5. 验收标准

- `make_tensor_descriptor_2d` 新增 `valid_inner/valid_outer`(支持 int 与 SSA),默认行为不变(回归通过)。
- microbench 给出 (a)~(e) 明确结论并回写 `05_risks_and_validation.md`。
- 提供一个最小 demo:tile 超 bound 时 load 补零 / store 丢弃,数值正确。

## 6. 交付物

- flydsl patch(`tdm_ops.py`)。
- microbench 脚本 + 结论。
- 在 `05` 回写 (a) load OOB 语义、(b) store OOB 语义、(c) 参考系结论。
