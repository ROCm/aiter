# push-grouped dispatch → 连续 GEMM1(gfx1250 fixed-slot)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 gfx1250 的 `dispatch_combine_v2` + a8w4 TDM GEMM1 上,把 token 分组从「GEMM1 消费端 pull-gather」挪到「dispatch 生产端 push 落地」(fixed-slot),使 GEMM1 连续读 A、wave-spec 保持 ON,消除 T-B 的 +27% 回归。

**Architecture:** dispatch 落地行号由「per-PE 单计数器」改为「per-(dest_pe, local_expert) 原子游标」,token 落地即按 expert 连续;新增 finalize 小核由计数产 GEMM1 tile 元数据 + 尾行 padding;GEMM1 在开关下跳过 `topids_to_rows`/`contiguous_psum_remap`/`a1_payload` 物化,直接连续读分组 recv 区。dispatch 与 GEMM1 仍是两次独立 launch(stream 有序),**不引入任何跨 kernel 就绪协议**。

**Tech Stack:** Python 3, PyTorch, aiter FlyDSL(`flyc.jit`/`flyc.kernel`),mori.cco 对称内存(`cco.Window(handle).lsa_ptr(pe, off)`),`flydsl_prims`(`atomic_add_global` / system store / fence / spin),gfx1250 WMMA a8w4(fp8 激活 + MXFP4 权重)。

**Spec:** `docs/superpowers/specs/2026-07-31-push-group-dispatch-gemm1-gfx1250-design.md`

## Global Constraints

- 目标硬件仅 gfx1250;其它硬件走原路径不受影响。
- 新 env 门控 `AITER_EP_PUSH_GROUP`(默认关);`=0` 必须与当前主线逐 kernel / 逐字节一致。与 `AITER_EP_FP8_TRANSPORT`(基座,需先开)组合。
- 只作用于 a8w4(`data_format=="a8w4"`,`quant_mode=="fp8"`)。
- **落地公式(全程统一)**:`local_expert = dest_expert - dest_pe*experts_per_rank`;`off = atomic_add_global(pg_running(dest_pe)+local_expert*4, 1)`;`grouped_row = local_expert*CAP + off`;publish 门 `off < CAP`。
- `CAP = align_up(push_group_cap, tile_m)`,默认上界 `world_size*max_num_inp_token_per_rank`(env `AITER_EP_PUSH_GROUP_CAP` 覆盖)。
- **不引入就绪协议**:两次独立 launch,GEMM1 启动时 dispatch+finalize 已完成;禁用 per-tile/per-expert ready 门控。
- padding 行 srcmap 填 sentinel `world_size*max_num_inp_token_per_rank`(GEMM1 照算、combine 依 sentinel 跳过),保 byte-exact。
- CUDA graph:固定 grid + 固定 kernel 序列;`pg_running` 每 launch 前 memset 清零(进 graph)。
- 测试入口:`op_tests/multigpu_tests/test_mega_moe.py`,**非 `/app` 目录**运行(避免 `/app/triton` 遮蔽);判据 `--acc_verify 1` 打印 `MEGA-CHECK PASS`。单 GPU 门禁在 `op_tests/flydsl_tests/`。
- 原语铁律:`aiter/ops/flydsl/dispatch_combine_v2/flydsl_prims.py::atomic_add_global`(fetch-and-add,已在现网 dispatch 用 8 处)、`store_i32_system`、`fence_system_{release,acquire}`;**禁止** flydsl 命名屏障。

## 已落地前提(勿重做)

- **T-A(DONE,-20%)**:dispatch 只传 fp8+e8m0,量化融进 rmsnorm(`AITER_EP_FP8_TRANSPORT=1`)。
- **push 骨架已验证**:`intranode_kernels.py::make_dispatch` Phase-1 已用 `atomic_add_global(peer_tok_off,1)`(`:192`)+ `window.lsa_ptr(dest_pe, off_out_tok)` push;唯一「未分组」点是 `off_tok_off` 为 per-PE 单计数器。
- **跨卡读写实测(spec §3.5)**:pull/push 数据面差 <5%;跨卡握手 15µs/RTT ⇒ 选 fixed-slot(握手最少)。

---

## Task 1:config 开关 + region + reset/ptrs(纯 Python,无 GPU)

**目的**:落地 fixed-slot 所需的 arena 布局与开关,默认关时零改动。

**Files:**
- Modify: `aiter/aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py`(`EpDispatchCombineConfig` 属性 + 模块级 `_push_group_regions` + `EpDispatchCombineOp` 的 regions/ptrs/reset)
- Test: `aiter/op_tests/flydsl_tests/test_push_group_regions.py`(新建,纯函数)

**Interfaces:**
- Produces:
  - `EpDispatchCombineConfig.push_group: bool`(env `AITER_EP_PUSH_GROUP`)
  - `EpDispatchCombineConfig.push_group_cap: int`(env `AITER_EP_PUSH_GROUP_CAP`,默认 `world_size*max_num_inp_token_per_rank`,再 `align_up(tile_m)`)
  - 模块级 `_push_group_regions(cfg, tile_m:int) -> list[tuple[str,int]]`(关时 `[]`;开时含 `("pg_running", num_local_experts*4)`,`num_local_experts = num_experts_per_rank`)
  - `EpDispatchCombineOp.push_group_ptrs() -> dict[str,int]`(各 region i64 指针,至少 `"pg_running"` 及 grouped recv token/scale 区)
  - `EpDispatchCombineOp.reset_push_group() -> None`(`pg_running` 及 finalize 计数 `.zero_()`)

- [ ] **Step 1: 写失败测试**

新建 `aiter/op_tests/flydsl_tests/test_push_group_regions.py`:

```python
import torch
from aiter.ops.flydsl.dispatch_combine_v2.dispatch_combine_op import (
    EpDispatchCombineConfig, _push_group_regions,
)

def _cfg(**ov):
    base = dict(rank=0, world_size=2, hidden_dim=512, max_num_inp_token_per_rank=128,
                num_experts_per_rank=4, num_experts_per_token=2, data_type=torch.float8_e4m3fn)
    base.update(ov); return EpDispatchCombineConfig(**base)

def test_push_group_defaults_off(monkeypatch):
    monkeypatch.delenv("AITER_EP_PUSH_GROUP", raising=False)
    assert _cfg().push_group is False
    assert _push_group_regions(_cfg(), tile_m=64) == []

def test_push_group_cap_default_and_align(monkeypatch):
    monkeypatch.setenv("AITER_EP_PUSH_GROUP", "1")
    monkeypatch.delenv("AITER_EP_PUSH_GROUP_CAP", raising=False)
    cfg = _cfg()  # world_size*mtpr = 2*128 = 256, align_up(tile_m=64) -> 256
    assert cfg.push_group is True
    assert cfg.push_group_cap % 64 == 0
    assert cfg.push_group_cap >= 2 * 128

def test_push_group_running_region(monkeypatch):
    monkeypatch.setenv("AITER_EP_PUSH_GROUP", "1")
    regs = dict(_push_group_regions(_cfg(), tile_m=64))
    assert regs["pg_running"] == 4 * 4  # num_experts_per_rank * 4 bytes
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl_tests/test_push_group_regions.py -v`
Expected: FAIL —— `cannot import name '_push_group_regions'`。

- [ ] **Step 3: 实现 config 属性 + region 纯函数 + ptrs/reset**

`EpDispatchCombineConfig` 属性区加(仿其现有 `os.environ.get` 风格):

```python
    @property
    def push_group(self) -> bool:
        return os.environ.get("AITER_EP_PUSH_GROUP", "0") in ("1", "true", "True", "yes", "on")

    @property
    def push_group_cap(self) -> int:
        import math
        tile_m = 64  # gemm1 a8w4 sort_block_m; 若 cfg 已有 tile_m 属性则改用之
        base = int(os.environ.get(
            "AITER_EP_PUSH_GROUP_CAP", str(self.world_size * self.max_num_inp_token_per_rank)))
        return ((base + tile_m - 1) // tile_m) * tile_m
```

模块级加:

```python
def _push_group_regions(cfg, tile_m):
    if not cfg.push_group:
        return []
    num_local_experts = cfg.num_experts_per_rank
    return [("pg_running", num_local_experts * 4)]
```

`EpDispatchCombineOp.__init__` 的 regions 列表构造后追加 `regions += _push_group_regions(self.cfg, tile_m)`(`tile_m` 取 a8w4 gemm1 sort_block_m=64);**并在 push_group 开时把 recv token / out_scales region 尺寸由 `max_recv` 改为 `num_local_experts*CAP`**(见 §约束;沿用现 region 命名,仅改 nbytes)。加 `push_group_ptrs()`(`self.arena.local_ptr(name)` 转 int)与 `reset_push_group()`(对 `pg_running` `.zero_()`)。

- [ ] **Step 4: 运行确认通过 + 提交**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl_tests/test_push_group_regions.py -v`
Expected: PASS

```bash
cd /app/aiter && git add aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py op_tests/flydsl_tests/test_push_group_regions.py
git commit -m "feat(ep): push-group config gate + pg_running region + reset/ptrs (default off)"
```

---

## Task 2:dispatch 落地改 per-expert 游标(byte-exact grouped)

**目的**:`make_dispatch` 在开关下把落地行号从 per-PE recv slot 换成 fixed-slot `grouped_row`,token/scale/wts/idx/tis 全按 `grouped_row` 落地。

**Files:**
- Modify: `aiter/aiter/ops/flydsl/dispatch_combine_v2/intranode_kernels.py::make_dispatch`(新增 `push_group`/`cap`/`experts_per_rank`/`off_pg_running` 形参 + Phase-1 落地寻址分支)
- Modify: `aiter/aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py`(dispatch 调用处透传 push_group 参数 + launch 前 `reset_push_group()`)
- Test: `aiter/op_tests/flydsl_tests/test_push_group_dispatch.py`

**Interfaces:**
- Consumes: Task 1 的 `push_group_ptrs()["pg_running"]`、`push_group_cap`。
- Produces: 开关开时 dispatch 后,recv token 区按 `[num_local_experts, CAP]` 布局,`grouped_row=local_expert*CAP+off` 处的 fp8 token/scale/tis 与该 route 源逐字节一致。

- [ ] **Step 1: 写失败测试(单 GPU / 双 rank grouped 落地 parity)**

新建 `aiter/op_tests/flydsl_tests/test_push_group_dispatch.py`:构造已知 `topk_ids`(每 expert 计数已知),`AITER_EP_PUSH_GROUP=1` 跑 dispatch,断言:对每个本地 expert `le`,`recv_tok[le*CAP : le*CAP+count[le]]` 的行集合 == 该 expert 所有源 route 的 fp8 token(按到达序,集合逐字节相等);`pg_running[le] == count[le]`。

```python
import os, pytest, torch
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs gfx1250")

def test_push_group_lands_grouped_byte_exact(monkeypatch):
    monkeypatch.setenv("AITER_EP_FP8_TRANSPORT", "1")
    monkeypatch.setenv("AITER_EP_PUSH_GROUP", "1")
    # 构造 EpDispatchCombineOp(world_size=1 自环 或 2),已知 topk_ids;
    # 跑 op.dispatch(...);读 pg_running 与 recv 区,按 le*CAP 切片比对源 token 集合。
    # 断言 running==期望计数,且每 expert 分组行逐字节匹配(atol=0)。
    ...
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl_tests/test_push_group_dispatch.py -v`
Expected: FAIL —— dispatch 未接受 `push_group`,落地仍为到达序。

- [ ] **Step 3: 实现落地寻址分支**

`make_dispatch` 加形参 `push_group=False, cap=0, experts_per_rank=0, off_pg_running=0`。Phase-1(`intranode_kernels.py:156-297`)在 `const_expr(push_group)` 开时,把现 recv-slot 分配(`:188-201` 的 `dest_tok_lane0 = atomic_add_global(peer_tok_off,1)` + `dest_tok_id`)替换为:

```
# lane 0:
local_expert = dest_expert - dest_pe * experts_per_rank
pg_addr = fx.Int64(window.lsa_ptr(dest_pe, off_pg_running)) + fx.Int64(local_expert)*fx.Int64(4)
off = P.atomic_add_global(pg_addr, fx.Int32(1))
grouped_row = local_expert * cap + off
# publish 门:no_dup & (off < cap)
```

其余不变:把现所有用 `dest_tok_id` 作为落地行号处(token embedding `:266-297`、scale `:243-261`、wts/idx `:221-241`、tis `:205-219`)改用 `grouped_row`;`is_dup_or_overflow` 的 overflow 判据由 `dest_tok_id >= max_recv` 改为 `off >= cap`。`replay` 路径缓存 `grouped_row`。dispatch_combine_op 调用处:开关开时透传 `push_group=True, cap=cfg.push_group_cap, experts_per_rank=cfg.num_experts_per_rank, off_pg_running=arena.offset("pg_running")`,并在每次 dispatch launch 前调 `self.reset_push_group()`。

- [ ] **Step 4: 运行确认通过**

Run: 同 Step 2。Expected: PASS(分组行逐字节等于源,running 计数正确)。

- [ ] **Step 5: 提交**

```bash
cd /app/aiter && git add aiter/ops/flydsl/dispatch_combine_v2/intranode_kernels.py aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py op_tests/flydsl_tests/test_push_group_dispatch.py
git commit -m "feat(ep): fixed-slot push-group dispatch landing (per-expert cursor, byte-exact)"
```

---

## Task 3:finalize 小核——由计数产 GEMM1 tile 元数据

**目的**:由 `pg_running[le]` 计数生成 a8w4 TDM GEMM1 所需的 tile 元数据 + 尾行 padding,替代 `topids_to_rows`/`contiguous_psum_remap`。

**Files:**
- Create: `aiter/aiter/ops/flydsl/kernels/push_group_finalize_gfx1250.py`(`@flyc.jit` finalize 核 + launcher)
- Test: `aiter/op_tests/flydsl_tests/test_push_group_finalize.py`

**Interfaces:**
- Consumes: `pg_running`(int32[num_local_experts])、`cap`、`tile_m`、`num_local_experts`、`rank`、`experts_per_rank`。
- Produces: `launch_push_group_finalize(pg_running_ptr, out_ptrs, num_local_experts, cap, tile_m, rank, experts_per_rank, stream)`,写出:
  - `pg_tile_row_base: int32[max_tiles]` —— 每 tile 在 `[num_local_experts, CAP]` grouped recv 里的起始行 = `le*cap + t*tile_m`
  - `pg_expert_ids: int32[max_tiles]` —— 每 tile 的 GLOBAL expert id = `rank*experts_per_rank + le`
  - `pg_num_valid: int32[1]` —— `sum_le(ceil(count[le]/tile_m))*tile_m`
  - `pg_srcmap padding`:每 expert `[count, num_tiles*tile_m)` 行填 sentinel `world_size*max_tok_per_rank`

- [ ] **Step 1: 写失败测试(元数据 == 现路径基准)**

新建 `aiter/op_tests/flydsl_tests/test_push_group_finalize.py`:给定 `pg_running`(随机每 expert 计数),跑 finalize,断言 `pg_tile_row_base`/`pg_expert_ids`/`pg_num_valid` 与 host 端参考实现(numpy 前缀和:每 expert `ceil(count/tile_m)` tiles、`row_base=le*cap+t*tile_m`、`gid=rank*epr+le`、`num_valid=sum*tile_m`)逐元素相等。

```python
import pytest, torch
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs gfx1250")

def test_finalize_metadata_matches_ref():
    from aiter.ops.flydsl.kernels.push_group_finalize_gfx1250 import launch_push_group_finalize
    tile_m, cap, epr, rank, ws, mtpr = 64, 256, 4, 0, 2, 128
    running = torch.tensor([70, 0, 130, 64], dtype=torch.int32, device="cuda")
    # ref: tiles per expert = ceil(count/tile_m) = [2,0,3,1]; num_valid=(2+0+3+1)*64=384
    # 跑 launch_push_group_finalize(...),读回三张表比对
    ...
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl_tests/test_push_group_finalize.py -v`
Expected: FAIL —— `launch_push_group_finalize` 不存在。

- [ ] **Step 3: 实现 finalize 核**

`push_group_finalize_gfx1250.py`:脚手架照抄 `crosscta_ring_spike_gfx1250.py` 的 `@flyc.kernel`+`@flyc.jit run().launch(grid=(1,1,1))`。device body 仿 megamoe `emit_direct_fixed_slot_finalize`(`mega_moe/dispatch.py:182-256`)+ `_wave_inclusive_scan_i32`(`mega_moe/dispatch.py:56-68`,可直接搬):warp0 内 `count[le]=load(pg_running,le)`、`num_tiles=ceil(count,tile_m)`、`inclusive_scan(num_tiles)` 得 `metadata_base`、`total_tiles`;每 le 写 `num_tiles` 条 `pg_expert_ids=rank*epr+le`、`pg_tile_row_base=le*cap+t*tile_m`;`pg_num_valid=total_tiles*tile_m`;尾行 `[count, num_tiles*tile_m)` 写 srcmap sentinel。**无跨 rank/跨 CTA store**。

- [ ] **Step 4: 运行确认通过 + 提交**

Run: 同 Step 2。Expected: PASS。

```bash
cd /app/aiter && git add aiter/ops/flydsl/kernels/push_group_finalize_gfx1250.py op_tests/flydsl_tests/test_push_group_finalize.py
git commit -m "feat(ep): push-group finalize kernel (counts -> tile_row_base/expert_ids/num_valid + pad)"
```

---

## Task 4:GEMM1 连续 A-load(开关下端到端 byte-exact)

**目的**:a8w4 TDM GEMM1 在开关下跳过 `topids_to_rows`/`contiguous_psum_remap`/`flydsl_moe_fused_quant_preshuffle`,直接吃 grouped recv + finalize 元数据,A-load 走连续块、wave-spec ON。

**Files:**
- Modify: `aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py::_grouped_a8w4_tdm_moe`(`_is_ep` 分支:`if cfg.push_group:` 走新元数据源)
- Modify: `aiter/aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py::launch_gemm_a8w4_tdm`(加 `push_group` const + grouped recv 指针形参;A-load prologue 连续块路径,wave-spec ON)
- Modify: `aiter/aiter/ops/flydsl/batched_gemm_mxfp4.py::flydsl_grouped_gemm_a8w4_masked`(host 透传)
- Test: `aiter/op_tests/flydsl_tests/test_push_group_gemm1_parity.py`

**Interfaces:**
- Consumes: Task 2 的 grouped recv(fp8 `[num_local_experts*CAP, K]` + e8m0 scale)、Task 3 的 `pg_tile_row_base`/`pg_expert_ids`/`pg_num_valid`。
- Produces: `flydsl_grouped_gemm_a8w4_masked(..., push_group=1, pg_recv=, pg_recv_scale=, tile_row_base=, expert_ids=, num_valid=)`:GEMM1 logits 与现 pull 路径逐字节一致。

- [ ] **Step 1: 写失败测试(端到端单 GPU parity)**

`test_push_group_gemm1_parity.py`:同一 a8w4 no-bias MoE,`AITER_EP_PUSH_GROUP={0,1}`(都开 `AITER_EP_FP8_TRANSPORT`),断言 GEMM1 logits **逐字节相同**(`atol=0,rtol=0`)—— push-group 连续路径 == 现 gather 路径。

- [ ] **Step 2: 运行确认失败**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl_tests/test_push_group_gemm1_parity.py -v`
Expected: FAIL —— `flydsl_grouped_gemm_a8w4_masked` 无 `push_group` 参数 / 输出不一致。

- [ ] **Step 3: 实现消费端连续 A-load**

`grouped_moe_gfx1250.py::_grouped_a8w4_tdm_moe`:`if _is_ep and cfg.push_group:` 跳过 `flydsl_moe_topids_to_rows` / `contiguous_psum_remap` / `flydsl_moe_fused_quant_preshuffle`(`:485-560`),改为:调 `launch_push_group_finalize`(Task 3)拿 `pg_tile_row_base`/`pg_expert_ids`/`pg_num_valid`,A/scale 源 = `push_group_ptrs()` 的 grouped recv 区,透传给 gemm1。`launch_gemm_a8w4_tdm`:`const_expr(push_group)` 开时 A-load 从 `pg_tile_row_base[m_tile]` 起**连续读** `[tile_m,K]`(复用 T-B.3 的连续 `add_tdm_loads`,**wave-spec 默认 ON** —— grouped 槽是连续块,无 T-B.3 非均匀 gather 计数问题),scale 同法;`expert_ids`→w1 local index(减 `rank*epr`)。`batched_gemm_mxfp4.py` host 透传(`push_group=0` 时全默认,向后兼容)。

- [ ] **Step 4: 运行确认通过 + 提交**

Run: 同 Step 2。Expected: PASS(逐字节)。

```bash
cd /app/aiter && git add aiter/ops/flydsl/grouped_moe_gfx1250.py aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py aiter/ops/flydsl/batched_gemm_mxfp4.py op_tests/flydsl_tests/test_push_group_gemm1_parity.py
git commit -m "feat(ep): GEMM1 push-group contiguous A-load (skip gather/materialize, wave-spec on)"
```

---

## Task 5:EP mega 端到端正确性 + 性能验收 + fallback

**Files:** Test: `op_tests/multigpu_tests/test_mega_moe.py`(复用)

- [ ] **Step 1: 正确性(push_group on)**

Run:
```bash
cd /tmp && AITER_EP_FP8_TRANSPORT=1 AITER_EP_PUSH_GROUP=1 \
ENABLE_CK=0 AITER_FORCE_A8W4=1 AITER_USE_GROUPED_GEMM=1 AITER_BF16_FP8_MOE_BOUND=0 \
FLYDSL_GPU_ARCH=gfx1250 torchrun --standalone --nproc_per_node=4 \
  /app/aiter/op_tests/multigpu_tests/test_mega_moe.py \
  -q a8w4_mxfp4 -e 384 -k 6 -hd 7168 -id 3072 \
  --combine scatter_fused --layers 2 --acc_verify 1
```
Expected: 2-rank 与 4-rank 均 `MEGA-CHECK PASS`,logits 与基线同量级。

- [ ] **Step 2: 回归(push_group off)**

Run: 同上 `AITER_EP_PUSH_GROUP=0`。Expected: `MEGA-CHECK PASS`,与主线逐 kernel 一致。

- [ ] **Step 3: 性能 A/B(真实维度)**

Run: on/off 各 `--profile_table 1 --layers 61`。核对:(1) `topids_to_rows` / `contiguous_psum_remap` / 独立 gather + `a1_payload`(`contiguous_m×hidden` fp8)物化消失;(2) GEMM1 device time 因 wave-spec 恢复 + 无 gather **不劣于甚至优于** T-B.4(对比小配置 +27% 回归是否消除);(3) 整层 device time 相对 T-A 基座再降或持平。写回 spec §7 结果区。

- [ ] **Step 4: 记录 + 提交**

```bash
cd /app/aiter && git add docs/superpowers/specs/2026-07-31-push-group-dispatch-gemm1-gfx1250-design.md docs/superpowers/plans/2026-07-31-push-group-dispatch-gemm1-gfx1250.md
git commit -m "docs(ep): record push-group dispatch->gemm1 correctness + perf"
```

**Fallback**(spec §6/§9):
- push-group 无净收益(fixed-slot 显存/带宽吃掉收益,或连续 A-load 未回收 wave-spec 损失)→ 保留 `AITER_EP_PUSH_GROUP` 默认关,记录实测;pull 路径(主线)与 `2026-07-31-crossCTA-ring` plan 仍是并行退路。
- finalize 元数据与现路径不一致(padding/sentinel 语义错)→ 回退「push 落地 grouped + 仍用现 `contiguous_psum_remap` 读计数产元数据」的混合折中,隔离风险。
- **期二/期三**(compact 密排省显存、跨 CTA 单巨核重叠)本 plan 不展开,见 spec §6。

---

## Self-Review

**Spec coverage**:
- spec §4 手术点1(dispatch 落地)→ Task 2 ✓;手术点2(finalize)→ Task 3 ✓;手术点3(GEMM1 连续 A-load)→ Task 4 ✓。
- spec §5 约束(CAP/env/overflow/padding/CUDA graph)→ Global Constraints 逐条 + Task 1(config/reset)/Task 2(overflow)/Task 3(padding)✓。
- spec §3.5 实测选型依据 → Global Constraints「不引入就绪协议」+ Task 结构(两次独立 launch)✓。
- spec §6 分期/fallback → Task 5 Fallback ✓。

**Placeholder scan**:config/region/测试给了完整 Python;kernel device body 标注确切参考函数(`emit_direct_fixed_slot_finalize` / `_wave_inclusive_scan_i32` / 连续 `add_tdm_loads`)、确切落地公式与确切原语(`atomic_add_global` / `store_i32_system`),执行者按名复刻,非 TBD(与现网 `crossCTA-ring` plan 同粒度)。

**Type consistency**:`push_group`(bool)/`push_group_cap`(int)/`_push_group_regions`/`push_group_ptrs`/`reset_push_group` 全程同名;region `pg_running`、元数据 `pg_tile_row_base`/`pg_expert_ids`/`pg_num_valid`、落地公式 `local_expert*CAP+off`、sentinel `world_size*max_tok_per_rank` 在 Task 1→2→3→4 统一;原语名与 `flydsl_prims.py` 对齐。
