# push-group persistent GEMM1 调度（gfx1250）

## 目标

push-group 的 fixed-slot 布局已通过紧凑 preshuffle 与紧凑 metadata grid 消除了大部分 CAP padding 成本，但 GEMM1 仍使用一个由静态上界驱动的 grid：

```text
bound_tiles = min(E_local * CAP / tile_m,
                  ceil(max_land / tile_m) + min(E_local, max_land))
```

低 batch 下此上界仍显著大于实际有效 M tile。例如 DeepSeek 形状（E=384、world=2、topk=6、CAP=64）在 bs=8 时，`max_land=96` 导致上界为 98 个 M tile；实际随机路由通常只形成约 48 个有效 expert tile。

本设计让 push-group **GEMM1** 以固定 worker grid 在设备端读取 finalize 产生的精确有效 tile 数，仅执行有效 tile。目标是进一步降低低 bs GEMM1 调度成本，同时保持 CUDA Graph capture、fixed-slot 缓冲布局、GEMM2 P2P scatter 与 pull 路径不变。

## 范围

包含：

- `grouped_a8w4_tdm_moe_push_scatter()` 内 push-group GEMM1 的 persistent grid-stride 调度；
- TDM a8w4 GEMM launch ABI 与 kernel body 的调度分支；
- GEMM1 persistent 与现有静态紧凑 grid 的 parity、ISA 与 2-rank graph e2e 验证。

不包含：

- GEMM2 persistent 调度；
- host 读回 `num_valid`、host-driven dynamic launch、device-side indirect launch；
- 跨 CTA 原子 work queue、planner/dispatch/GEMM 单巨核；
- dispatch、fixed-slot recv buffer、preshuffle 数据格式或 P2P combine 协议的变更。

## 当前数据流

```text
dispatch
  -> pg_running[E_local]
  -> finalize
       -> tile_row_base / expert_ids / tile_valid（紧凑有效 M tile）
       -> num_valid（有效行数，tile_m 对齐）
  -> GEMM1（当前：static bound_tiles × n_tiles grid）
  -> a2 payload/scale（仍按 fixed-slot row 写入）
  -> GEMM2 static compact grid + P2P scatter
  -> combine
```

`tile_row_base[m_tile]` 是 fixed-slot recv 行号，而非紧凑 logical M 行。GEMM1 的 `expert_ids[m_tile]` 与 `tile_valid[m_tile]` 同样由 finalize 密排。故只要 persistent scheduler 枚举完整的 `[0, valid_m_tiles)` metadata 索引，A load、scale load、GEMM1 output 和下游 GEMM2 均无需重排。

## 调度设计

### Device scalar

复用 finalize 当前写入的 `(1,) int32 num_valid`：

```text
valid_m_tiles = num_valid[0] / tile_m
total_work    = valid_m_tiles * n_tiles
```

`num_valid` 已是每 expert `ceil(count/tile_m) * tile_m` 之和，因而一定可被 `tile_m` 整除。它在 dispatch/finalize/GEMM1 的同一 stream 顺序上被生产和消费，不需要 host 同步。

### Persistent work assignment

GEMM1 固定 launch：

```text
grid.x = persistent_workers       # 默认 get_cu_num()
grid.y = 1
```

workgroup `worker_id` 在 kernel 内处理：

```text
for work_id = worker_id; work_id < total_work; work_id += persistent_workers:
    run_tile(work_id, valid_m_tiles)
```

`run_tile()` 复用既有 DeepGEMM contiguous-M swizzle，只将：

- `bid_x` 替换为 `work_id`；
- `total_m_tiles = ceil(i32_m / tile_m)` 替换为 `valid_m_tiles`。

它继续得到 `(m_tile, n_tile)`，并通过当前 metadata 访问：

```text
blk_m  = tile_row_base[m_tile]
expert = expert_ids[m_tile]
mn_oob = tile_valid[m_tile]
```

无效尾部不会进入 loop，因此不会读取 sentinel metadata 或执行 WMMA。

### 运行时开关

push GEMM1 默认采用 hybrid 策略：环境请求 persistent 后，只有当前静态紧凑 grid 仍包含上界尾部时才启用：

```text
persistent = AITER_EP_PUSH_GROUP_PERSISTENT_GEMM1 != 0 and grid_m < cm
```

`grid_m < cm` 完全由静态 `max_land`、`E_local`、`CAP` 决定，不读取 `num_valid` 到 host；bs=8 满足该条件，static grid 已饱和的 512/2048 档不满足。设置为 `0` 强制回退到提交 `ac9f2edb7` 中的静态紧凑 grid。可选：

```text
AITER_EP_PUSH_GROUP_PERSISTENT_WORKERS=<int>
```

用于 worker sweep；缺省取 `get_cu_num()`。开关仅影响 push GEMM1；pull 与 GEMM2 保持现状。

## 代码修改

### `mxfp4_preshuffle_gfx1250_tdm.py`

1. `launch_gemm_a8w4_tdm()` 新增：
   - `ep_persistent_gemm1: Constexpr[int]`
   - `persistent_workers: Constexpr[int]`
   - `arg_num_valid_rows: fx.Pointer`
2. cache key 纳入 persistent flag 与 worker 数，避免 fixed 与 persistent binary 复用。
3. 抽取当前 swizzle→A/B DMA→WMMA→epilogue 主体为 `_run_tile(work_id, total_m_tiles)`。
4. fixed 路径保留当前 `m_tiles * n_tiles` launch 与一次 `_run_tile(bid_x, m_tiles)`。
5. persistent 路径用 FlyDSL `scf.ForOp` 建立 grid-stride loop，读取 `arg_num_valid_rows[0]` 并调用 `_run_tile`。
6. persistent launch grid 为 `(persistent_workers, 1, 1)`；固定路径不变。
7. `num_valid == 0` 时 loop 不执行，所有 CTA 安全返回。

### `batched_gemm_mxfp4.py`

`flydsl_grouped_gemm_a8w4_masked()` 新增：

- `ep_persistent_gemm1=False`
- `num_valid_rows=None`
- `persistent_workers=None`

关闭时传 ABI dummy pointer；开启时透传 `num_valid` device tensor 和 worker 数。该 wrapper 的其他调用者默认不变。

### `grouped_moe_gfx1250.py`

`grouped_a8w4_tdm_moe_push_scatter()`：

1. 从环境读取 persistent flag 与 workers；
2. GEMM1 调用传入现有 `num_valid`；
3. GEMM1 persistent 默认开启；
4. GEMM2 仍使用 `contiguous_m=grid_m` 的静态紧凑 grid，且 `ep_p2p_write`、`pg_rowmap` 代码不变。

`num_valid` 当前在 graph capture 内每层由 finalize 重新写入，因此 persistent path 保持 graph-safe。

## 正确性与性能风险

### 正确性

- `num_valid` 与 `trb/eids/tvd` 必须来自同一 finalize invocation；
- persistent path 只遍历 `[0, valid_m_tiles)`，保证每个有效 tile 恰好一次；
- A2 fixed-slot 输出地址由 `tile_row_base` 决定，不能替换为 logical `m_tile*tile_m`；
- `num_valid=0`、单 expert、每 expert 一行和 CAP 饱和均需覆盖；
- GEMM2 未改动，确保 P2P scatter rowmap ABI 不受 persistent scheduler 影响。

### 性能

persistent loop 可能增加 VGPR/SGPR live range、spill 或降低 occupancy。它应在 bs=8 减少 CTA 数，但大 bs 的循环化调度可能劣于大量独立 CTA。默认开启不代表接受大 bs 回归；若性能 gate 失败，应保留实现但切换为 shape-based hybrid policy。

## 验证

1. **Finalize 单测**
   - 保持 fixed 与 compact metadata 测试；
   - 增加零 count、单 expert、稀疏单行、CAP 饱和。

2. **GEMM1 parity**
   - 扩展 `test_push_group_gemm1_parity.py`；
   - 对相同 fixed-slot payload、scale、`trb/eids/tvd/num_valid`，比较静态紧凑与 persistent 的 GEMM1 output、A2 payload 和 A2 scale。

3. **ISA**
   - 固定/persistent 分别 dump final ISA；
   - 记录 VGPR、SGPR、VGPR spill、SGPR spill、occupancy；
   - 若出现明显 occupancy cliff，停止默认开启并改用 hybrid 策略。

4. **CUDA Graph 2-rank e2e**
   - `test_mega_moe.py`，DeepSeek shape `hd=7168,id=3072,E=384,k=6,world=2`；
   - bs `8/64/128/256/512/2048`，pull/static-push/persistent-push；
   - 每配置至少 11 个独立 graph replay 样本并报告 median/p50；
   - 所有配置 `MEGA-CHECK PASS`，logits_diff 与 pull 同量级。

## 验收标准

- persistent GEMM1 在 bs=8 不慢于当前静态紧凑 push，并使端到端 push 不慢于 pull 的测量误差范围；
- bs≥64 不吞掉当前 push 收益；若任一形状的 median 回归超过 2%，实施 hybrid 启用条件而非保留“始终开启”；
- graph capture/replay 与所有 2-rank 正确性检查通过；
- `AITER_EP_PUSH_GROUP_PERSISTENT_GEMM1=0` 精确回退当前静态紧凑行为。
