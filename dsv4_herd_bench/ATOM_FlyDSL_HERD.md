# ATOM 接入 FlyDSL HERD（DeepSeek-V4-Flash decode）

对应 patch：`atom_flydsl_herd.patch`（只改 `atom/model_ops/moe.py`，+113 / −15）。

## 1. 结论

- ATOM 原来没有 FlyDSL HERD 的接入：FlyDSL 路径一直按标准 Top-6 路由。patch 让 decode 阶段改为输出 Top-7 候选，并把 `herd_topk=6` 交给 aiter 的 `fused_moe`；"丢弃一个最冷门候选 + 重新归一化 + 乘 routed scale"在 gfx1250 grouped MoE 的 route/prefix-sum kernel 里融合完成。
- 路由结果和 aiter 自带的 `herd_topk_gating` 完全一致（专家 id 100% 相同，权重 bit-exact）。
- 64 并发下（热缓存），FlyDSL 开 HERD 后 Mean TPOT 从 9.82 ms 降到 8.89–9.11 ms（约 −7% ~ −9.5%），和 Triton+HERD 的 8.57 ms 只差约 4–6%；gsm8k 精度和 Triton+HERD 在误差范围内。

## 2. 环境

| 项目 | 版本 |
|---|---|
| 容器 | `gcy_atom`（`rocm/fw-bringup-mlperf:gfx1250-atom-20261016`） |
| GPU | 1 × gfx1250（机器只有一张卡，所以全部用 `-tp 1`） |
| ATOM | `/app/ATOM`，分支 `r1_1012_rebased`，`f19778df4` |
| aiter | `/home/cguo/aiter`（editable 安装），分支 `cguo/gfx1250_herd`，`38ab715d4` |
| FlyDSL | `0.3.3.dev886`（容器自带，未替换） |
| 模型 | `/data/DeepSeek-V4-Flash`：E=256，topk=6，sqrtsoftplus，`norm_topk_prob=True`，`routed_scaling_factor=1.5`，前 3 层 hash routing |

## 3. 开关语义（改动前就存在，容易踩坑）

`Mxfp4MoEMethod.__init__` 里：

```python
use_triton_moe = envs.ATOM_USE_TRITON_MOE            # 显式设置时
self.use_triton = use_triton_moe and not ep_moe
self.use_triton_decode = (self.use_triton or self.use_triton_ep) and envs.ATOM_USE_TRITON_MOE_DECODE
```

以及 `apply()` 里 `use_triton_now = self.use_triton`，只有 `use_triton_decode` 打开时才改成"仅 decode"。所以：

| `ATOM_USE_TRITON_MOE` | `ATOM_USE_TRITON_MOE_DECODE` | prefill | decode |
|---|---|---|---|
| 0 | 任意 | FlyDSL | FlyDSL（DECODE 开关无效） |
| 1 | 1 | FlyDSL | Triton/gluon |
| 1 | 0 | Triton | Triton |

- 原 `serve_dsv4.sh` 写的是 `ATOM_USE_TRITON_MOE=0` + `ATOM_USE_TRITON_MOE_DECODE=1`，实际上跑的是纯 FlyDSL，`AITER_TRITON_USE_HERD=1` 也不会生效。
- 本次测试用的配置：

| 模式 | 环境变量 |
|---|---|
| Triton+HERD | `ATOM_USE_TRITON_MOE=1 ATOM_USE_TRITON_MOE_DECODE=1 AITER_TRITON_USE_HERD=1` |
| FlyDSL+HERD | `ATOM_USE_TRITON_MOE=0 ATOM_USE_TRITON_MOE_DECODE=0 AITER_FLYDSL_USE_HERD=1` |
| FlyDSL | `ATOM_USE_TRITON_MOE=0 ATOM_USE_TRITON_MOE_DECODE=0 AITER_FLYDSL_USE_HERD=0` |

其余环境变量和 `serve_dsv4.sh` 相同。

## 4. aiter 侧提供的接口（`cguo/gfx1250_herd`）

| 接口 | 作用 |
|---|---|
| `fused_moe(..., herd_topk=K, herd_routed_scaling_factor=s)` | `topk_ids/topk_weights` 传 `[T, K+1]` 候选；gfx1250 grouped MoE 用 `herd_route_psum_remap` 替换普通的 route/psum，一个 kernel 里完成：批内候选热度统计 → 每个 token 丢掉热度最低（并列时分数最低）的候选 → 剩下 K 个归一化后乘 `s` → 生成 `topids_to_rows`/`psum`/`gather_w` |
| `grouped_gemm_gfx1250_a8w4` 的约束 | `herd_topk` 要求：没有 `expert_mask`（非 EP）、没有 `stage2_scatter`、`doweight_stage1=False`、没有 MegaMoE dispatch context，且 `topk_ids.shape[1] == K+1` |
| `topk_gating`（`AITER_FLYDSL_USE_HERD=1` 时） | 当 16 ≤ M ≤ 128 且命中 profile（DSV4-Flash：sqrtsoftplus、E=256、K=6）时，内部直接调用 `herd_topk_gating`，输出已经丢弃过的 `[T, K]`。aiter 拿不到 prefill/decode 信息，所以只按 M 判断 |

HERD 的批大小窗口是 `AITER_FLYDSL_HERD_MIN_M/MAX_M`，默认继承 `AITER_TRITON_HERD_MIN_M/MAX_M`（16 / 128），和 Triton HERD 相同。

## 5. ATOM 改动明细

都在 `atom/model_ops/moe.py` 里。

### 5.1 导入 aiter 的 HERD 开关和窗口

```python
from aiter.ops.topk import _FLYDSL_HERD_MAX_M, _FLYDSL_HERD_MIN_M, _FLYDSL_USE_HERD
```

直接复用 aiter 的值，保证 ATOM 和 aiter 用的是同一个开关（`AITER_FLYDSL_USE_HERD`）和同一个 M 窗口，不会出现两边判断不一致。这三个名字在 aiter main 和容器原来的 aiter 里都存在，所以模块级导入不会因为 aiter 版本而失败。

### 5.2 `Mxfp4MoEMethod.__init__`：新增 `self.use_flydsl_herd`

```python
self.use_flydsl_herd = (
    _FLYDSL_USE_HERD
    and self.is_gfx1250
    and self.is_guinterleave
    and not ep_moe
    and not getattr(get_current_atom_config(), "eplb_enable", False)
)
```

| 条件 | 原因 |
|---|---|
| `is_gfx1250 and is_guinterleave` | `herd_topk` 只在 gfx1250 grouped a8w4 路径（GUGU / `GateMode.INTERLEAVE`）实现，其他路径 aiter 会直接报错 |
| `not ep_moe` | 融合 drop 的 route kernel 不接受 `expert_mask` |
| `not eplb_enable` | EPLB 需要 logical→physical 映射和负载记录（`select_experts_with_record`），HERD 分支绕过了它 |

这个 flag 放在 EPLB 强制关闭 Triton 的代码块之后。

### 5.3 `apply()` 的 FlyDSL 尾部：按步决定是否走 HERD

原来是无条件调用 `self.select_experts_with_record(...)` 得到 `[T, 6]`；现在先调用 `_flydsl_herd_step(...)`：

- **命中时**：
  ```python
  topk_ids     = torch.empty((T, K + 1), int32)
  topk_weights = torch.empty((T, K + 1), float32)
  topk_gating(topk_weights, topk_ids, router_logits, e_score_correction_bias,
              False, 1.0, score_func=scoring_func)
  ...
  moe_extra_args["herd_topk"] = K
  moe_extra_args["herd_routed_scaling_factor"] = layer.routed_scaling_factor
  ```
  - `need_renorm=False, routed_scaling_factor=1.0`：HIP `topk_gating` 按 `score + bias` 选 Top-7，输出的是**未加 bias、未归一化**的原始 sqrtsoftplus 分数。这正是 `herd_route_psum_remap` 期望的 `cand_scores`，也是 aiter 自己在 `herd_topk_gating` 非融合分支里生成候选的方式。
  - `topk_gating` 收到宽度 7 时不命中 aiter 内部的 HERD profile（profile 要求 topk=6），所以会直接走普通 kernel，不会被二次 drop。
  - `herd_routed_scaling_factor = layer.routed_scaling_factor`：和 `select_experts` 对 sqrtsoftplus 的约定一致（`route_scale = routed_scaling_factor`，在路由里乘进去）。
  - `herd_topk` 只在命中时加进 `moe_extra_args`，所以不命中的调用和旧版 aiter 都不会收到这个参数。
- **不命中时**：保持原来的 `select_experts_with_record(...)`，行为不变。

### 5.4 新增 `_flydsl_herd_step()`：逐步门控

```python
if not self.use_flydsl_herd or self.fused_experts is not None: return False
if not _FLYDSL_HERD_MIN_M <= M <= _FLYDSL_HERD_MAX_M:          return False
if (use_grouped_topk or not renormalize or custom_routing_function is not None
        or scoring_func != "sqrtsoftplus" or apply_router_weight_on_input
        or layer.num_fused_shared_experts > 0 or layer.expert_mask is not None):
    return False
ctx = get_forward_context().context
return ctx is not None and not ctx.is_prefill and ctx.running_tokens_are_unified
```

| 条件 | 原因 |
|---|---|
| `fused_experts is None` | 只接直接调用 `fused_moe` 的分支；modular kernel（EP transport）分支不传 `herd_topk` |
| M 在 [16, 128] | 和 aiter / Triton HERD 用同一个窗口：M 太小时专家并集缩不了，M 太大时属于 prefill 规模 |
| `not use_grouped_topk` | 分组 top-k 的候选语义不同 |
| `renormalize` 且 `sqrtsoftplus` | 融合 drop 总是做归一化并乘 scale，这是 sqrtsoftplus 的约定；sigmoid（MiniMax）的 scale 在 MoE 外面乘，softmax 语义也不同，所以没有接 |
| `custom_routing_function is None` | 前 3 层 hash 层查 `tid2eid` 表，不参与 HERD |
| `not apply_router_weight_on_input` | aiter 的 `herd_topk` 不支持 `doweight_stage1` |
| `num_fused_shared_experts == 0` | 融合进来的 shared expert 会占一列 topk，drop 时可能把它丢掉。DSV4-Flash 的 shared expert 是 FP8、routed 是 FP4，dtype 不同所以本来就不融合 |
| `expert_mask is None` | 同 5.2（非 EP） |
| `not is_prefill and running_tokens_are_unified` | HERD 只用于 decode；和 Triton decode 分流用同一个判据，在 DP attention + TP MoE 场景下保证各 rank 对同一步的路由方式一致，否则 TP reduce 会把用不同路由算出的 partial 结果加在一起 |

CUDA graph：`apply()` 运行在 `torch.ops.aiter.moe_forward` custom op 里，这个判断在每个 batch size 的 graph capture 时执行一次并固化进 graph；K+1 候选 buffer 从 graph 内存池里分配。

### 5.5 一次性诊断日志

```python
@lru_cache(maxsize=None)
def _log_flydsl_herd_batch(num_tokens, top_k): logger.info("FlyDSL HERD decode routing: ...")
```

每个 batch size 只打一次，用来确认 HERD 确实生效。实测服务日志里 16/32/48/64/128 五个 capture size 都有这条日志。不需要的话可以删掉。

## 6. 数据流对比

| 阶段 | 改动前（FlyDSL decode） | 改动后（FlyDSL decode，16 ≤ M ≤ 128） |
|---|---|---|
| 路由 | `topk_gating` → `[M, 6]`，已归一化 × 1.5 | `topk_gating` → `[M, 7]` 原始分数 |
| route/psum | `flydsl_moe_topids_to_rows` + psum | `herd_route_psum_remap`：一个 kernel 里统计热度、drop、归一化 × 1.5，并生成 rows/psum/gather_w |
| GEMM1/GEMM2/gather-reduce | 按 6 个路由计算 | 按 6 个路由计算，但涉及的专家更少 |

在随机输入上，激活的专家数（top-6 → HERD）：M=16 时约 77 → 72，M=32 时约 133 → 116，M=64 时约 188 → 153，M=128 时约 236 → 193。

## 7. 验证

### 7.1 路由级校验（`check_herd_routing.py`）

随机 bf16 logits、fp32 bias，M ∈ {16, 32, 48, 64, 128}，每个 M 跑 5 次。把 ATOM 新路径（Top-7 → `herd_route_psum_remap`）和下面两者比较：
- aiter 的 `herd_topk_gating`（融合 sqrtsoftplus HERD selector）：专家 id 一致率 100%，权重差 0（bit-exact）；
- torch 参考（同一组 7 个候选做热度/分数 drop）：专家 id 一致率 100%，权重最大误差 ≤ 9e-8；
- `masked_m.sum() == M × 6`，每一行都恰好落在它所属专家的连续区间里。

### 7.2 端到端

- 服务日志：5 个 HERD capture size 都打出了诊断日志；不开 HERD 时没有。
- Sanity：32 个并发的贪心解码请求（`17*23=391`、`Paris`、`3+5-2=6`、`buenos días`），所有配置的回答都正确。
- gsm8k（`eval_dsv4.sh`，3-shot，1319 题）：

| 配置 | flexible-extract | strict-match |
|---|---|---|
| Triton+HERD | 0.9492 ± 0.0060 | 0.9484 ± 0.0061 |
| FlyDSL+HERD | 0.9515 ± 0.0059 | 0.9522 ± 0.0059 |

### 7.3 性能（`bench_dsv4.sh`：ISL=1024、OSL=1024、64 并发、256 条请求、TP1、热缓存）

| 配置 | 输出吞吐 (tok/s) | Mean TPOT (ms) | Median TPOT (ms) | Mean TTFT (ms) | P99 ITL (ms) |
|---|---|---|---|---|---|
| Triton+HERD | 6798.26 | 8.57 | 8.51 | 867.99 | 8.93 |
| FlyDSL+HERD（run2） | 6619.96 | 8.89 | 8.88 | 795.13 | 9.34 |
| FlyDSL+HERD（run3） | 6469.90 | 9.11 | 9.12 | 805.60 | 9.69 |
| FlyDSL（不开 HERD） | 6041.32 | 9.82 | 9.78 | 796.32 | 10.32 |

- 最早一轮（run1）bench 期间在做首次 JIT 编译（aiter 的 `module_top_k_per_row`、FlyDSL kernel emit），TTFT 和 P99 ITL 都被拉高（Triton+HERD 4991.76 tok/s，FlyDSL+HERD 4246.51 tok/s），这组数作废。
- 两次 FlyDSL+HERD 相差约 2.3%，其他配置各只跑了一次；Triton 和 FlyDSL 之间 4–6% 的差距需要多跑几次才能定论。
- 这个差距里约一半来自前 3 层 hash 层：Triton decode 路径在这几层用的不是模型本来的路由（见第 8 节第 4 点），所以并不完全是同一工作量下的对比。差距的完整拆分见 `E2E_vs_UnitTest_HERD.md`。

## 8. 已知问题和限制

1. **prefill 小 batch 也可能被 aiter 内部 HERD 处理。** `AITER_FLYDSL_USE_HERD=1` 时，aiter 的 `topk_gating` 只按 M 判断，ATOM 不命中 HERD 的步（prefill）走 `select_experts → topk_gating([T, 6])`，如果 16 ≤ T ≤ 128，会在 aiter 内部做非融合的 HERD drop。本次计时的 bench 和 gsm8k 的 prefill 都远大于 128 token，只有 run3 计时前的一条 sanity 请求（18 token）落在窗口里，所以测得的数据不受影响。要做到严格只在 decode 生效，可以让 aiter 的 `topk_gating` 增加一个关闭内部 HERD 的参数，或者在 `use_flydsl_herd` 打开时让 ATOM 的 prefill 直接调用 `topk_gating_fwd`。
2. **从 aiter 导入了私有名字**（`_FLYDSL_USE_HERD` 等）。如果 aiter 改名会导致导入失败；更稳妥的做法是 aiter 提供一个公开的 helper。
3. **只接了 DSV4 这一类配置**：sqrtsoftplus + renormalize、TP、非融合 shared expert。EP / MegaMoE / modular kernel、融合 shared expert、sigmoid / softmax 模型都没有接。
4. **已有问题，不是本 patch 引入的**：ATOM 的 Triton decode 分支调用 aiter `routing()` 时没有使用 `custom_routing_function`，所以 Triton+HERD 下前 3 层 hash 层在 decode 阶段用的是 gate top-k 再加 HERD，而不是 `tid2eid` 查表。两种配置在这 3 层上算的不是同一件事。
   - 按 e2e trace 逐层拆分，FlyDSL 在这 3 层的 GEMM1 约 87 µs、GEMM2 约 49 µs，Triton 是 46–59 / 24–31 µs。每个 step 合计约 +140–160 µs，大约是两者 e2e 差距的一半。
   - 这部分不是 FlyDSL kernel 慢，详见 `E2E_vs_UnitTest_HERD.md` 的 4.5 和 5.5 节。
5. **没有 `fused_moe(herd_topk)` 的数值单测**（需要构造 gfx1250 GUGU 布局的 MXFP4 权重）。目前的正确性来自：路由级逐位校验 + 下游 GEMM/gather 路径和非 HERD 共用 + 端到端精度。
6. **没有比较"aiter `topk_gating` 内部 HERD（不改 ATOM）"和"融合 drop"两种方案的性能。** 选融合 drop 是为了能按 phase 门控，并复用 grouped MoE 本来就要跑的 route/psum kernel。
7. patch 目前只存在于容器 `/app/ATOM` 里，没有 commit。

## 9. 使用方法

```bash
# 在 ATOM r1_1012_rebased (f19778df4) 上
cd /app/ATOM && git apply /home/cguo/aiter/dsv4_herd_bench/atom_flydsl_herd.patch

# 起服务 / 跑 bench / 跑精度（容器内）
cd /home/cguo/aiter/dsv4_herd_bench
MOE_MODE=flydsl_herd bash serve_dsv4_tp1_herd.sh /data/DeepSeek-V4-Flash                # 只起服务
TASKS=bench bash run_mode.sh flydsl_herd                                                # 起服务 + bench + 关服务
TASKS=eval  bash run_mode.sh flydsl_herd                                                # 起服务 + gsm8k + 关服务
python check_herd_routing.py                                                            # 路由级校验
```

`MOE_MODE` 可选 `triton_herd` / `flydsl_herd` / `flydsl`。`run_mode.sh` 的日志默认写到脚本目录下的 `logs/`，可以用 `LOG_DIR` 改。本文引用的历史日志（run1 冷缓存、gsm8k、`run2/`、`run3/`）在 `/home/cguo/dsv4_herd_bench/logs/`。
