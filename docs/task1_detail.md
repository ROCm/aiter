# OPUS GEMM split-K workspace 代码变更记录

本文件只记录实际发生的代码变更：改了哪些文件，以及每个文件新增、替换、删除了
哪些内容。计划中的文件在真正修改前不写入完成记录。

若需要直接比较最初 `ca68b4f...` 的 C++ 隐式 workspace流程与当前 Torch 管理流程，并查看
完整45个代码文件的端点增删清单，而不需要逐 Step 历史，请直接阅读
`docs/opus_gemm_splitk_workspace_torch_current_flow_changes.md`。该文档的“原流程”不是中间
`WorkspacePlan` 版本。

记录规则：

- 按 Step、提交或明确标记的未提交工作树追加，不覆盖已经完成的历史记录；
- 每个文件记录状态（新增/修改/删除）、增删行数、修改位置和具体内容；
- 位置优先写函数、类、常量等稳定符号，行号只作为对应提交的快照；
- 同时写清明确保留、没有修改的相关文件，防止后续步骤提前删除；
- 2026-08-11 的未提交结构精简与 #4246 fused 迁入记录位于文件末尾；最后一节是当前权威
  状态，并取代旧记录中把 `WorkspacePlan`/两个独立 workspace Python 模块描述为现行架构，
  或把 1378 个 fused kid描述为“尚未合入”的表述；历史提交内容保留用于审计。
- 最新权威结果：gfx950 external workspace 48 个且全部 FP32；gfx1250 two-stage 496 个且全部
  BF16，fused 1378 个且为 BF16 780 / FP32 598；CPU过滤回归为
  `151 passed, 18 deselected, 0 failed`，其中18项未执行而不是失败；gfx950 focused suite为
  `166 passed, 14 skipped, 0 failed`；gfx950 140-kid sweep最终为
  `140 passed, 0 failed`，原10个 mono-tile non-workspace FP32失败已经定位并修复；有效
  workspace性能 A/B也已完成。文件最末“任务一最终续测”一节是最新测试状态，早先冻结、
  失败和性能暂停段落保留为历史审计。

## Step 1：让 Python 在分配前知道最终 kid 和 split-K

- 日期：`2026-08-10`
- 分支：`splitk_to_torch_2`
- 修改前基线：`ca68b4f3501762c15c550cb920a5516e9710cf89`
- 完成提交：`83ce59db94debab2d14a46a88ed9d57e6d2bf0ae`
- 提交标题：`[OPUS] Resolve a16w16 dispatch in Python`
- 合计：10 个文件，新增 1330 行，删除 160 行
- ABI：本步没有修改 workspace ABI

### Step 1 文件总表

| 状态 | 文件 | 行数变化 |
|---|---|---:|
| 修改 | `csrc/opus_gemm/opus_gemm_common.py` | `+53/-0` |
| 修改 | `csrc/opus_gemm/opus_gemm_tune.py` | `+2/-12` |
| 修改 | `csrc/opus_gemm/codegen/gen_instances_gfx942.py` | `+6/-5` |
| 修改 | `aiter/ops/opus/gemm_op_a16w16.py` | `+122/-143` |
| 新增 | `aiter/ops/opus/_selector_a16w16.py` | `+382/-0` |
| 新增 | `aiter/ops/opus/heuristics/__init__.py` | `+40/-0` |
| 新增 | `aiter/ops/opus/heuristics/a16w16_gfx1250.py` | `+34/-0` |
| 新增 | `aiter/ops/opus/heuristics/a16w16_gfx950.py` | `+44/-0` |
| 新增 | `aiter/ops/opus/heuristics/a16w16_gfx942.py` | `+259/-0` |
| 新增 | `op_tests/test_opus_dispatch.py` | `+388/-0` |

`csrc/opus_gemm/codegen/gen_instances_gfx942.py` 不在最初列出的文件清单中，但实际提交
修改了它，因此在这里如实记录。

### 1. `csrc/opus_gemm/opus_gemm_common.py`

状态：修改，`+53/-0`。

#### 修改位置一：gfx942 instance 常量区（提交后第 520-524 行）

新增内容：

```python
GFX942_BF16WS_EXACT_N = frozenset({64, 128, 256, 384, 512, 1024, 2048})
```

用途：

- 作为 gfx942 bf16-workspace exact-N 的共享权威集合；
- 供 runtime selector、tuner 和 gfx942 codegen 共同使用；
- 只共享合法 N，不把 codegen 专属的 `(VEC, N_VEC, ROWS_PER_BLOCK)` 配置投影到
  runtime。

#### 修改位置二：`heuristic_kids_for_arch()` 后（提交后第 1201-1245 行）

新增 `get_kernel_instance(arch, family, kid)`：

- 从现有 `kernels_list` 返回 canonical `OpusGemmInstance`；
- 将 `arch`、`family` 统一转为小写；
- 当前只允许 `family == "a16w16"`；
- 校验 kid 能转成整数、instance 存在、`kernel_tag` 属于 a16w16；
- 校验 instance 的 architecture 与请求 architecture 一致；
- 查询失败返回 `None`。

新增 `kernel_needs_external_workspace(arch, family, kid)`：

- 先调用 `get_kernel_instance()` 做 arch/family/kid 窄查询；
- 未知 kernel 抛 `KeyError`，不把未知 kid 当成“不需要 workspace”；
- 已知 kernel 通过现有 `SPLITK_KIDS` 判断是否需要外部 workspace；
- 没有新建第二份数值区间表。

删除内容：无。

明确没有新增 `_catalog.py`，也没有生成 frozen metadata 投影。

### 2. `csrc/opus_gemm/opus_gemm_tune.py`

状态：修改，`+2/-12`。

#### 修改位置一：`opus_gemm_common` import 区（提交后第 48-72 行）

新增导入：

```python
GFX942_BF16WS_EXACT_N
```

#### 修改位置二：tune-time 常量区（原文件约第 98-108 行）

完整删除本文件私有副本：

```python
BF16WS_EXACT_REDUCE_SHAPES = (
    ...
)
```

该元组原来重复维护 exact-N 以及部分 reduce row-count 信息。Step 1 删除这份 N 集合
副本，合法 N 改为消费 common 中的共享常量；其他 tuner 自己需要的候选过滤和 tune
规则仍保留。

#### 修改位置三：`kid_rejects_shape()`（提交后第 405-409 行）

替换前：

```python
return not any(N == n_exact for n_exact, _ in BF16WS_EXACT_REDUCE_SHAPES)
```

替换后：

```python
return N not in GFX942_BF16WS_EXACT_N
```

### 3. `csrc/opus_gemm/codegen/gen_instances_gfx942.py`

状态：修改，`+6/-5`。

#### 修改位置一：import（提交后第 8 行）

替换前：

```python
from opus_gemm_common import OpusGemmInstance
```

替换后：

```python
from opus_gemm_common import GFX942_BF16WS_EXACT_N, OpusGemmInstance
```

#### 修改位置二：`EXACT_N_ROWBLOCK_REDUCE_CONFIGS` 后（提交后第 139-141 行）

新增一致性断言：

```python
assert frozenset(
    vec * nvec for vec, nvec, _ in EXACT_N_ROWBLOCK_REDUCE_CONFIGS
) == GFX942_BF16WS_EXACT_N
```

含义：codegen 继续拥有详细 row-block 配置，但这些配置推导出的 N 集合必须与 common
共享集合完全一致。

#### 修改位置三：gfx942 bf16ws launcher 条件生成（提交后第 327-331 行）

删除：在此处临时从 `EXACT_N_ROWBLOCK_REDUCE_CONFIGS` 再构造一份 N 集合。

替换为：

```python
exact_reduce_shape_conditions = " ||\n        ".join(
    f"(N == {n_exact})" for n_exact in sorted(GFX942_BF16WS_EXACT_N)
)
```

### 4. `aiter/ops/opus/gemm_op_a16w16.py`

状态：修改，`+122/-143`。

#### 修改位置一：模块说明和 import（提交后第 3-23 行）

替换模块说明：

- 删除旧说明中“CSV miss 进入 generic C++ bf16 selector”的流程；
- 改为说明固定流程
  `explicit -> tuned CSV -> per-arch heuristic -> framework fallback`；
- 说明 OPUS 生产路径统一通过 id-based `opus_gemm_a16w16_tune()`。

删除 import：

```python
import logging
from . import common as _opus_common
```

新增 import：

```python
from csrc.opus_gemm.opus_gemm_common import (
    get_kernel_instance,
    kernel_needs_external_workspace,
)
from ._selector_a16w16 import select_launch_config
```

新增 `_SUPPORTED_OPUS_ARCHES = ("gfx942", "gfx950", "gfx1250")`。

没有新增 `_catalog.py`，没有修改局部 `sys.path`。

#### 修改位置二：新增 `_device_arch_and_cu()`（提交后第 26-40 行）

新增行为：

- 从实际 tensor device 的 CUDA/HIP properties 读取 `gcnArchName`；
- 去掉 `gfx*:...` 中冒号后的 feature 后缀并转小写；
- properties 无有效 gfx 名时回退到 `get_gfx_runtime()`；
- 同时返回 `multi_processor_count` 作为 CU 数；
- 无法得到 architecture 时抛 `RuntimeError`。

#### 修改位置三：`opus_gemm_a16w16_tune()`（提交后第 155-233 行）

保留：

- 旧 positional-int 调用兼容；
- XQ/WQ/Y layout 检查；
- raw C++ binding 的原有返回约定。

删除旧行为：

- 删除 mono-tile N/K 不对齐时记录 warning 并调用
  `_opus_gemm_bf16_dispatch()` 重新选择 kernel 的分支。

新增/替换行为：

- 从 XQ/Y 得到 batch/M/N/K；
- 调用 `_device_arch_and_cu()`；
- 把传入 kid/split-K 作为 explicit 请求交给 `select_launch_config()` 严格解析；
- framework fallback 对 explicit 请求不合法，出现时抛 `RuntimeError`；
- raw binding 参数从原始 `kernelId/splitK` 改为
  `config.actual_kid/config.launch_split_k`。

#### 修改位置四：legacy generic binding 注释（提交后第 236-270 行）

`_opus_gemm_bf16_dispatch` binding 本身没有删除，签名仍保留；注释改为明确它只作
Step 1 parity golden probe，生产 a16w16 路径不再调用。

#### 修改位置五：`is_splitk_kid()`（提交后第 275-290 行）

完整删除：

```python
_SPLITK_KID_RANGES = (
    (200, 300),
    (1200, 1300),
    (10200, 10300),
    (20000, 21000),
)
```

替换 `is_splitk_kid()` 实现：

- 依次在 gfx942/gfx950/gfx1250 的 canonical a16w16 registry 查 kid；
- 找到 instance 后调用 `kernel_needs_external_workspace()`；
- 三架构都找不到时返回 `False`；
- 不再以整数范围判断 capability。

兼容常量 `_SPLITK_KID_MIN = 200` 和 `_SPLITK_KID_MAX = 299` 仍保留，但不再作为
workspace capability 权威。

#### 修改位置六：新增 `_framework_a16w16()`（提交后第 426-442 行）

新增内容：

- bf16 输出使用 `torch.bmm(XQ, WQ.transpose(1, 2))`；
- fp32 输出先把 XQ/WQ 转为 float 再做 `torch.bmm`；
- 支持 `[N]` 和 `[batch, N]` bias；
- 把结果转换为 Y dtype 后写入预分配的 `Y`。

#### 修改位置七：`gemm_a16w16_opus()`（提交后第 445-519 行）

删除旧控制流：

- 高层函数内单独处理 explicit-kid 分支；
- 高层函数直接调用 `_opus_common.lookup_tuned()` 并拆出 `solidx/splitK`；
- CSV miss 调用 `_opus_gemm_bf16_dispatch()`；
- 关于 generic C++ selector 处理 CSV miss 的旧注释。

新增/替换控制流：

- reshape/validate 后读取 arch 和 CU 数；
- 统一调用 `select_launch_config()`；
- selector 返回 framework fallback 时调用 `_framework_a16w16()`；
- explicit、tuned CSV、Python heuristic 三种 OPUS 结果统一调用
  `opus_gemm_a16w16_tune()`；
- 传入的值为最终 `actual_kid` 和 `launch_split_k`；
- low-level tune wrapper 将其作为 explicit 请求再次做 launch 边界校验，但不会重新
  查询 tuned CSV；
- CSV miss 的正常生产路径不再调用 generic C++ bf16 selector。

### 5. `aiter/ops/opus/_selector_a16w16.py`

状态：新增文件，`+382/-0`。

#### 新增 `LaunchConfig`（第 38-77 行）

字段：

- `arch`、`family`、`source`；
- `requested_kid`、`actual_kid`；
- `requested_split_k`、`allocation_split_k`、`launch_split_k`；
- gfx942 使用的 `effective_split_k`；
- framework fallback 使用的 `fallback_reason`。

另有兼容 property：`kid`、`split_k`、`is_framework_fallback`。

#### 新增 framework fallback 构造（第 80-91 行）

`_framework_fallback()` 创建 kid/split-K 均为空或 0 的 `LaunchConfig`，并保存原因。

#### 新增 dtype 和 shape compatibility 校验（第 94-140 行）

- `_output_dtype_name()` 统一 bf16/fp32 名称；
- `_instance_output_compatible()` 区分两阶段 workspace kernel 与 direct-output
  kernel；
- `_instance_shape_compatible()` 拒绝 gfx1250 batch != 1；
- mono-tile 校验 N/K tile 对齐；
- non-OOB instance 校验 M/N tile 对齐。

#### 新增 gfx942 requested kid -> actual kid 解析（第 143-156 行）

非 exact-N 时：

- `10210 -> 10200`；
- `10213 -> 10203`；
- `10216` 抛 `ValueError`；
- 其他 kid 保持不变。

exact-N 时 requested kid 保持不变。

#### 新增 `_build_launch_config()`（第 159-256 行）

新增处理：

- 把 kid/split-K 转为整数；
- 校验 requested instance；
- 解析 actual kid 后重新查询 actual instance；
- 按 actual kid 查询 workspace capability；
- 校验 shape、output dtype 和 bias capability；
- 暂时拒绝 gfx942 split-K + bias 进入现有 tune C++ ABI；
- workspace kernel 计算 allocation split-K；
- gfx942 调用独立 split resolver 得到 allocation/effective/launch split-K；
- 返回完全解析后的 `LaunchConfig`。

#### 新增 `select_launch_config()`（第 259-379 行）

新增固定选择顺序：

1. explicit override；
2. tuned CSV；
3. 对应 architecture 的 Python heuristic；
4. framework fallback。

具体规则：

- explicit 最优先，失败直接报错，不查询 tuned CSV；
- tuned lookup 采用 lazy import；
- tuned row 的 kid、split-K、arch、shape、dtype 或 bias 任一无效时，整对
  `(kid, splitK)` 原子丢弃；
- heuristic kid 必须属于 `HEURISTIC_DEFAULT_KIDS_BY_ARCH`；
- unsupported arch 返回 framework fallback；
- heuristic 选出的 kernel 因当前 raw ABI 无法使用时返回 framework fallback。

### 6. `aiter/ops/opus/heuristics/__init__.py`

状态：新增文件，`+40/-0`。

新增内容：

- 导入三架构的 `select_kid`；
- 新增 `A16W16_HEURISTICS` 映射：
  - `gfx942 -> select_kid_gfx942`；
  - `gfx950 -> select_kid_gfx950`；
  - `gfx1250 -> select_kid_gfx1250`；
- 新增 family-local `select_kid()`，按 architecture 分派；
- 未知 architecture 抛 `ValueError`；
- 新增对应 `__all__`。

该目录只实现 a16w16 selector，不是所有 OPUS family 的统一 selector。

### 7. `aiter/ops/opus/heuristics/a16w16_gfx1250.py`

状态：新增文件，`+34/-0`。

新增 `select_kid()`（第 6-31 行），等价移植 gfx1250 C++ heuristic：

- `M % 32 == 0 && N % 128 == 0 -> 20007`；
- `M % 32 == 0 && N % 64 == 0 -> 20006`；
- `M % 32 == 0 && N % 32 == 0 -> 20005`；
- `N % 128 == 0 -> 20004`；
- `N % 64 == 0 -> 20003`；
- 其余 -> `20000`。

### 8. `aiter/ops/opus/heuristics/a16w16_gfx950.py`

状态：新增文件，`+44/-0`。

新增 `select_kid()`（第 6-41 行），等价移植 gfx950 C++ heuristic，包含：

- `M <= 4`、`M <= 64`、`M <= 128` 三段 small-M 分支；
- M/N/K 对齐时选择 non-OOB mirror；
- `N % 16 == 0`、`K % 64 == 0` 且 loop 为偶数时允许 split-barrier；
- bias 请求避开不支持 bias 的 split-barrier 分支；
- 返回 kid 集合：`208`、`1208`、`206`、`1206`、`200`、`1200`、`300`、
  `1300`。

### 9. `aiter/ops/opus/heuristics/a16w16_gfx942.py`

状态：新增文件，`+259/-0`。

#### 新增 launcher symbol -> kid 反查（第 31-57 行）

- 遍历 `HEURISTIC_DEFAULT_KIDS_GFX942`；
- 通过 `get_kernel_instance("gfx942", "a16w16", kid)` 取得 canonical instance；
- 使用现有 `instance.name` 建立 `_SYMBOL_TO_KID`；
- 检查重复 symbol 和未知 symbol；
- 没有手写第二张 launcher-symbol/kid 表。

#### 新增 gfx942 heuristic 分支（第 59-165 行）

- `_split_barrier_ok()`：移植 split-barrier eligibility；
- `_bf16ws_band()`：移植 bf16-workspace shape band；
- `_select_bf16_symbol()`：按原 C++ 顺序移植 K=4096、WKC small-M/N、bf16ws、
  N=384、large-N、split-barrier 和末端 fallback 分支；
- `select_kid()`：bf16/no-bias 使用完整特化 heuristic，fp32 或 bias 使用原通用分支；
- heuristic 先返回 launcher symbol，再通过 canonical instance name 解析成 kid。

#### 新增 gfx942 split-K resolver（第 168-251 行）

新增 `SplitKResolution`：

- `requested`：原始 split-K；
- `allocation`：workspace 分配安全上界；
- `effective`：launcher 实际使用值。

新增 `resolve_split_k()`：

- 显式 split-K > 0 时，allocation 先保留调用者请求；
- split-K == 0 时按 tile 数、CU 数计算 auto split-K；
- auto split-K 限制在 `[1, 16]`；
- 保留当前 launcher 的 `kernel_tag.endswith("_p1")` target-WG 判定；
- 计算 K 方向总 iteration，要求每个 split 至少 2 个 iteration；
- 对指定 kernel tag 要求 full/last split 均为偶数 loop；
- 不满足 iteration/parity 时逐一向下 clamp；
- 最终仍不满足 even-loop 时抛 `ValueError`。

### 10. `op_tests/test_opus_dispatch.py`

状态：新增文件，`+388/-0`。

新增 `_csv_miss()` 和 `_select()` 测试 helper。

新增测试内容：

- `get_kernel_instance()` 的 arch/family 隔离；
- `kernel_needs_external_workspace()` 的已知/未知 kid 行为；
- gfx942 exact-N 共享集合；
- gfx1250 heuristic 六类分支；
- gfx950 small-M、non-OOB、split-barrier、bias 分支；
- gfx942 K=4096、WKC、bf16ws、N=384、large-N、fp32 和 bias 分支；
- gfx942 launcher symbol 从 `instance.name` 反查；
- gfx942 auto split-K 和 even-loop down-clamp；
- gfx942 显式 split-K > 16 的旧 launcher parity；
- K iteration 太少时拒绝；
- explicit 优先于 tuned CSV；
- wrong-arch explicit kid 严格失败；
- 合法 tuned `(kid, splitK)` 成对保留；
- wrong-arch/shape-invalid tuned row 原子丢弃；
- gfx942 `10210/10213` 非 exact-N redirect；
- gfx942 `10216` 非 exact-N 拒绝；
- exact-N 时 `10210/10213/10216` 保持原 kid；
- 非法 tuned `10216` 的 split-K 不泄漏到 heuristic 结果；
- unsupported arch framework fallback；
- heuristic kid 必须在 force-compiled 集合；
- gfx1250 batch != 1 拒绝；
- gfx942 split-K + bias 当前 ABI fallback；
- CSV miss 走 tune wrapper，并断言 `_opus_gemm_bf16_dispatch` 未被调用。

该测试文件执行结果：`56 passed`。

### Step 1 明确没有修改或删除的相关文件

| 文件/代码 | Step 1 状态 |
|---|---|
| `csrc/opus_gemm/opus_gemm.cu` | 未修改；旧 allocator/registry 未删除 |
| `csrc/opus_gemm/include/gfx1250/opus_gemm_heuristic_dispatch_gfx1250.cuh` | 未修改，保留作 parity golden |
| `csrc/opus_gemm/include/gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh` | 未修改，保留作 parity golden |
| `csrc/opus_gemm/include/gfx942/opus_gemm_heuristic_dispatch_gfx942.cuh` | 未修改，保留作 parity golden |
| `aiter/jit/build/module_deepgemm_opus/blob/opus_gemm_lookup.h` | 未主动删除；ignored build 生成文件 |
| `aiter/jit/build/module_deepgemm_opus/blob/opus_gemm_a16w16_tune_lookup.h` | 未主动删除；ignored build 生成文件 |
| `_opus_gemm_bf16_dispatch` | binding 保留，只退出生产控制流 |
| `opus_gemm_workspace_init()` | 保留，workspace ABI 尚未切换 |

## Step 2：增加 WorkspacePlan 和集中 Torch 分配

- 日期：`2026-08-10`
- 分支：`splitk_to_torch_2`
- 修改前基线：`83ce59db94debab2d14a46a88ed9d57e6d2bf0ae`
- 完成提交：`b2c99c8494ac1193c97ee1be23452a6e43af48d6`
- 提交标题：`[OPUS] Add typed a16w16 workspace plans`
- 合计：4 个代码文件，新增 788 行，删除 1 行
- ABI：本步没有修改 raw C++/pybind workspace ABI，新路径保持未启用

### Step 2 文件总表

| 状态 | 文件 | 行数变化 |
|---|---|---:|
| 新增 | `aiter/ops/opus/_workspace.py` | `+186/-0` |
| 新增 | `aiter/ops/opus/_workspace_a16w16.py` | `+141/-0` |
| 修改 | `aiter/ops/opus/gemm_op_a16w16.py` | `+83/-1` |
| 新增 | `op_tests/test_opus_workspace.py` | `+378/-0` |

### 1. `aiter/ops/opus/_workspace.py`

状态：新增文件，`+186/-0`。

#### 新增 `WorkspacePlan`

新增动态、不可变的调用级计划，字段为：

```text
shape
dtype
required_numel
alignment
```

构造时统一检查：

- shape extent、`required_numel` 和 alignment 均为正整数；
- alignment 是 2 的幂；
- shape 元素数及 dtype 对应字节数不超过 Torch 可表示上限；
- `required_numel` 不超过 plan shape 的分配容量。

#### 新增 `checked_numel()`

- 对任意正 extent 序列做逐项 checked multiply；
- 在乘法前检查调用方给定的上限，Python 大整数不会掩盖最终 Torch/C++ extent
  溢出；
- 不含任何 a16w16、architecture、kid 或 tile 规则。

#### 新增 `validate_workspace()`

共享验证覆盖：

- 必须是 Tensor；
- dtype 与 plan 完全相同；
- 可选的 expected device 完全相同；
- contiguous；
- `numel >= required_numel`；
- `data_ptr != 0` 且满足 plan alignment。

caller-owned workspace 不强制与 plan 使用相同维度，只要连续地址范围、类型、设备、
对齐和容量满足合同即可，因此 flat 或更大的 typed Tensor 可以安全复用。

#### 新增 `allocate_workspace()`

- 唯一分配表达式是
  `torch.empty(plan.shape, dtype=plan.dtype, device=device)`；
- 分配后立即复用 `validate_workspace()`；
- 没有 `cache`、`lru_cache` 或全局 Tensor 所有权。

### 2. `aiter/ops/opus/_workspace_a16w16.py`

状态：新增文件，`+141/-0`。

#### 新增 `plan_a16w16_workspace()`

输入为 canonical actual `OpusGemmInstance`、完整 `(arch, family, kid)` 身份、
`M/N/K/batch` 和 selector 已解析的 allocation split-K。

新增行为：

- 用 `get_kernel_instance()` 验证传入对象确为 actual kid 的 canonical instance；
- 用 `kernel_needs_external_workspace()` 判断 capability；已知 non-workspace kid 返回
  `None`，不按 tag 或 kid 数值区间猜测；
- 按 actual instance 的 `B_M/B_N` 计算 padded M/N；
- 按 actual instance 的 `splitk_workspace_dtype` 映射 bf16/fp32 typed workspace；
- gfx950、gfx942 生成 `[split_k, batch, padded_M, padded_N]`；
- gfx1250 生成 `[split_k, padded_M, padded_N]`，并拒绝 `batch != 1`；
- 用 actual instance 的 `B_K` 推导当前调用的 K-tile 上限，在分配前拒绝永远只会被
  launcher 向下 clamp 的超大 split-K；
- plan 使用 16-byte workspace base alignment，并通过共享 checked extent 计算精确
  `required_numel`。

没有新增 instance metadata 副本，也没有改动 `opus_gemm_common.py`。

### 3. `aiter/ops/opus/gemm_op_a16w16.py`

状态：修改，`+83/-1`。

#### 新增 `_prepare_a16w16_workspace()`

- 只接受已经完全解析的 `LaunchConfig`；
- 通过 `config.actual_kid` 重新取得 canonical instance；
- 调用 a16w16 planner；
- non-workspace plan 要求 caller workspace 也为 `None`；
- workspace plan 缺少显式 Tensor 时调用共享 `allocate_workspace()`；
- caller 显式传入 Tensor 时只调用共享 `validate_workspace()`，不再分配；
- 返回 `(WorkspacePlan | None, Tensor | None)` 供 Step 5 raw ABI 使用。

#### 新增 `_launch_a16w16_with_torch_workspace()`

准备完整的集中执行顺序：

```text
resolved LaunchConfig
  -> actual instance
  -> WorkspacePlan | None
  -> allocate or validate caller Tensor
  -> raw_launch(XQ, WQ, Y, bias, workspace, actual_kid, launch_split_k)
```

raw callable 采用注入参数，当前 legacy binding 没有被传入此 helper。注释明确 Step 5
更新 raw ABI 后才切换两个公共入口。

#### 明确保留的当前生产行为

- `_opus_gemm_a16w16_tune_raw` 的 schema 和调用参数未修改；
- `opus_gemm_a16w16_tune()` 仍调用 legacy raw ABI；
- `gemm_a16w16_opus()` 仍通过当前 tune wrapper 启动；
- 当前生产路径不创建未被 C++ 使用的 Torch workspace；
- `opus_gemm_workspace_init()` 和 C++ allocator/registry 尚未删除。

### 4. `op_tests/test_opus_workspace.py`

状态：新增文件，`+378/-0`。

新增 24 个 CPU-side 测试，覆盖：

- `WorkspacePlan` 动态合同及 checked extent overflow；
- typed shape/dtype 分配，并确认连续两次调用返回不同 Tensor、没有 Python cache；
- exact capacity 与 larger flat capacity 成功；
- 少 1 element、错 dtype、错 device、noncontiguous、错 alignment 失败；
- 当时 Step 2 基线中的 gfx950 fp32、gfx942 fp32/bf16、gfx1250 fp32 tile padding 和
  shape 规则（gfx1250 后续已由本日志末尾的 #4246 BF16 恢复取代）；
- 普通、persistent 和 atomic-accumulate kid 返回 `None`；
- gfx942 `10210 -> 10200` 后按 actual kid 生成 fp32 plan，传 requested instance 被拒绝；
- gfx1250 batch > 1 失败；
- split-K 超出 actual instance K-tile 上限时在分配前失败；
- 显式 workspace 复用共享验证且不调用 allocator；
- 准备的 Step 5 helper 把 typed workspace、actual kid 和 launch split-K 交给 fake raw
  callable；
- Step 2 当前生产路径没有提前启用 Torch workspace 分配。

测试结果：

```text
pytest -q op_tests/test_opus_workspace.py
24 passed

pytest -q op_tests/test_opus_dispatch.py op_tests/test_opus_workspace.py
80 passed
```

附加 registry 遍历：当前 canonical `SPLITK_KIDS` 中可由三架构 a16w16 窄查询取得的
552 个 workspace instance（gfx950 48、gfx942 8、gfx1250 496）均成功生成 dtype、
shape、`required_numel == allocation_numel` 自洽的计划。

### Step 2 明确没有修改或删除的相关文件

| 文件/代码 | Step 2 状态 |
|---|---|
| `csrc/opus_gemm/opus_gemm.cu` | 未修改；旧 allocator/registry 未删除 |
| `csrc/opus_gemm/gen_instances.py` | 未修改；Step 3 才拆 generated dispatch |
| 三架构 codegen/traits/pipeline/reduce | 未修改；Step 3/4 才切 direct pointer |
| raw C++/pybind `opus_gemm_a16w16_tune` | 未修改；Step 5 才增加 workspace 参数 |
| `opus_gemm_workspace_init()` | 保留；Step 5 才改 deprecated Python no-op |
| `_opus_gemm_bf16_dispatch` | 保留为 parity probe |

## Step 3：把 codegen 拆成 non-workspace / workspace 两套 dispatch

- 日期：`2026-08-10`
- 分支：`splitk_to_torch_2`
- 修改前基线：`b2c99c8494ac1193c97ee1be23452a6e43af48d6`
- 完成提交：`0a9bd8101c1d8ac84d6734deaf2ec385a45c0e54`
- 提交标题：`[OPUS] Split a16w16 workspace dispatch tables`
- 合计：8 个代码文件，新增 856 行，删除 689 行
- ABI 状态：generated host launcher 和 dispatch table 已拆成 5/6 参数两套；Step 4
  的 traits/pipeline/reduce 以及 Step 5 的 raw entry/pybind 尚未接通，因此本提交仍是
  计划内的 ABI 中间态，不作为最终可运行版本

### Step 3 文件总表

| 状态 | 文件 | 行数变化 |
|---|---|---:|
| 修改 | `csrc/opus_gemm/gen_instances.py` | `+147/-112` |
| 修改 | `csrc/opus_gemm/codegen/gen_instances_gfx950.py` | `+40/-53` |
| 修改 | `csrc/opus_gemm/codegen/gen_instances_gfx942.py` | `+49/-79` |
| 修改 | `csrc/opus_gemm/codegen/gen_instances_gfx1250.py` | `+47/-62` |
| 修改 | `csrc/opus_gemm/include/opus_gemm_common.cuh` | `+75/-0` |
| 修改 | `csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh` | `+138/-185` |
| 修改 | `csrc/opus_gemm/include/gfx942/opus_gemm_arch_gfx942.cuh` | `+197/-99` |
| 修改 | `csrc/opus_gemm/include/gfx1250/opus_gemm_arch_gfx1250.cuh` | `+163/-99` |

### 1. `csrc/opus_gemm/gen_instances.py`

状态：修改，`+147/-112`。

#### reduce ABI 和 split-K host ABI（提交后第 72-236 行）

- `SPLITK_REDUCE_ABI_MAP` 的 gfx950、gfx942、gfx1250 三个条目全部从
  `const opus_splitk_ws_handle*` 改为 `const void* ws_ptr`；
- `ws_type` 同步改为 `const void*`，使 generated forward declaration 和 reduce explicit
  instantiation 都采用 direct pointer ABI；
- `SPLITK_TAGS` 补入此前遗漏的
  `a16w16_em3en4_lds1_pgr2_sk`，使它不会误入 non-workspace 表；
- 保留原 `A16W16_TUNE_HOST_EXTRA` 作为 non-workspace 5 参数 ABI；
- 新增 `A16W16_WORKSPACE_TUNE_HOST_EXTRA`，按
  `XQ, WQ, Y, workspace, bias, splitK` 记录 split-K 6 参数显式实例化。

#### `gen_lookup_dict()`（提交后第 446-534 行）

把原 `(M,N,K) -> launcher function pointer` 表改为按架构、输出 dtype 分区的
`(M,N,K) -> integer kid` 表：

```text
GENERATE_OPUS_LOOKUP_TABLE_GFX950_BF16 / FP32
GENERATE_OPUS_LOOKUP_TABLE_GFX942_BF16 / FP32
GENERATE_OPUS_LOOKUP_TABLE_GFX1250_BF16 / FP32
```

每张表同时生成 `*_SIZE`：

- entry 只保存 shape 和 kid，不再引用 `&launcher<CTYPE>`；
- split-K tuned row 不再伪装成 5 参数 function pointer；
- 继续按 output dtype 分桶，同一 shape 的 bf16/fp32 winner 仍可不同；
- 继续按 lexicographic `(M,N,K)` 排序供 `lower_bound` 使用；
- 用 canonical `(arch, instance.name) -> kid` 反查保证 emitted kid 可追溯；
- 空表生成合法的空 macro 和 size 0，支持单架构/subset build。

#### `gen_a16w16_tune_lookup()`（提交后第 536-621 行）

原来同一张表混放所有 a16w16 launcher；现在每个架构生成三张类型严格的表：

```text
GENERATE_A16W16_TUNE_LOOKUP_GFX*_BF16(CTYPE)
GENERATE_A16W16_TUNE_LOOKUP_GFX*_FP32(CTYPE)
GENERATE_A16W16_WORKSPACE_TUNE_LOOKUP_GFX*
```

- 前两张只包含 non-workspace 5 参数 `OpusA16W16Kernel`；
- workspace 表只包含 split-K 6 参数 `OpusA16W16WorkspaceKernel`；
- workspace launcher 保持已有 `<fp32_t>` host specialization；
- 每张表生成独立 `*_SIZE`；
- 三架构严格分区，错误架构的同名/同值 kid 不会进入当前架构表；
- workspace instance 若没有 `fp32_t` host specialization，codegen 立即抛错；
- 所有 entry 继续按 kid 排序。

#### `gen_manifest_head()`（提交后第 660-725 行）

- non-workspace a16w16 manifest declaration 保持 5 参数；
- split-K manifest declaration 改为 6 参数并显式加入
  `aiter_tensor_t &workspace`；
- manifest 分类直接使用 `SPLITK_TAGS`，与生成 tune table 的分类一致；
- a8w8 scale/noscale manifest ABI 没有变化。

#### 明确保留的 subset-compile 逻辑

下列逻辑原样保留，没有为了拆表而删除：

- tuned CSV 的 `csv_kids` 收集；
- `compiled_kids_sidecar` 的读取和写回；
- `S = CSV | sidecar | HEURISTIC_DEFAULT_KIDS`；
- per-arch filter；
- heuristic fallback kid 必须在 compile set 中的 invariant；
- tuned CSV 只引用实际编译 kid 的过滤。

### 2. `csrc/opus_gemm/include/opus_gemm_common.cuh`

状态：修改，`+75/-0`。

在 host-only guard 内新增两个 family-neutral helper；没有加入 a16w16、architecture、
kid、tile 或 workspace dtype 特判。

#### `opus_checked_extent_product()`（提交后第 28-41 行）

- 接收动态 `initializer_list<size_t>`；
- 每个 extent 必须大于 0；
- 每次相乘前检查 `size_t` 上溢；
- 错误信息携带 launcher 传入的 label。

#### `opus_validate_workspace()`（提交后第 43-90 行）

共享验证覆盖：

- `required_numel > 0`；
- alignment 是非零 2 的幂；
- workspace device 与参考输入 XQ 一致；
- dtype 与 launcher 请求的 typed workspace 完全一致；
- contiguous；
- `workspace.numel() >= required_numel`；
- `data_ptr()` 非空；
- base address 满足 alignment；
- `required_numel * element_size` 也经过 checked `size_t` 乘法。

验证成功后返回 `workspace.data_ptr()` 的 direct pointer。a16w16 function pointer 类型和
dispatch 没有放入这个公共头，仍留在三个 arch adapter header。

### 3. `csrc/opus_gemm/codegen/gen_instances_gfx950.py`

状态：修改，`+40/-53`。

修改位置：`gen_flatmm_splitk_instance()`（提交后第 1222-1415 行）。

新增/替换：

- split-K launcher 增加 `aiter_tensor_t &workspace`；
- host explicit instantiation 改用
  `A16W16_WORKSPACE_TUNE_HOST_EXTRA`；
- include `opus_gemm_common.cuh` 以复用 checked extent/validator；
- 保留原 split-K 向下 clamp 和 `pfk * B_K` 最小 iteration 检查；
- clamp 后按 `size_t` 依次计算 `padded_M`、`padded_N`、单 slice numel 和
  `[split_k, batch, padded_M, padded_N]` 的 `required_numel`；
- 在写入 int kernel stride 前检查 padded extent 和 slice numel 不超过
  `INT_MAX`；
- 用共享 validator 要求 fp32、16-byte aligned workspace；
- validator 返回的 direct pointer 同时写入 `kargs.ptr_ws` 并传给 bf16/fp32、
  bias/no-bias 四条 reduce launch；
- `stride_ws_batch` 由 checked slice numel 转换。

删除：

- `hipStreamIsCapturing()`；
- `opus_splitk_ws_get()` registry lookup；
- cache grow 和 4 MiB grow rounding；
- launcher 内的 `hipMalloc()` / `hipFree()`；
- grow 前 `hipDeviceSynchronize()`；
- `opus_splitk_ws_handle` 读写。

main kernel grid、main launch、reduce grid和四条 reduce 数值路径均保留。

### 4. `csrc/opus_gemm/codegen/gen_instances_gfx942.py`

状态：修改，`+49/-79`。

#### reduce forward declaration / explicit instantiation（提交后第 144-182 行）

- baseline、bf16ws fallback 和 exact-N rowblock reduce 的 workspace 参数全部改为
  `const void*`；
- `EXACT_N_ROWBLOCK_REDUCE_CONFIGS` 导出的 N 集合继续断言严格等于共享
  `GFX942_BF16WS_EXACT_N`；
- 当前共享 exact-N 集合为 `{64, 128, 256, 384, 512, 1024, 2048}`。

#### `gen_splitk_gfx942_instance()`（提交后第 198-549 行）

- split-K launcher 增加 `aiter_tensor_t &workspace`，host explicit instantiation 改用
  6 参数 ABI；
- 从 actual instance 的 `splitk_workspace_dtype` 选择
  `AITER_DTYPE_bf16` 或 `AITER_DTYPE_fp32`；
- 删除 10210 和 10213 非 exact-N 时对 fp32-workspace launcher 的 host redirect；
- 所有 bf16-workspace launcher，包括 10210、10213、10216，统一从共享 exact-N
  集合生成 `AITER_CHECK`；非 exact-N 的 10216 继续硬失败；
- 保留 bf16-workspace 当前只允许 bf16 Y 的最后防线；
- auto split-K 的 tile product 改为 checked `size_t`，随后保留原 `[1,16]` clamp；
- 保留显式/auto split-K 语义、最少两次 K iteration 和 dbuf2 even-loop clamp；
- clamp 后 checked 计算 padded extent、slice numel 和
  `[split_k, batch, padded_M, padded_N]` required numel；
- 验证 actual kid 对应的 typed workspace 和 16-byte alignment；
- direct pointer 写入 `kargs.ptr_ws`，并传给 exact-N rowblock、bf16ws fallback 和
  fp32 baseline reduce；
- 原 main launch、exact-N fast path、generic fallback 和 bias分支均保留。

删除：

- capture 状态检查；
- host registry get/grow；
- `hipMalloc()` / `hipFree()` 和 device synchronize；
- host handle、device handle mirror、H2D mirror sync；
- 在 launcher 内取得 `opus_splitk_ws_device_handle()` 的路径。

正常 Python 路径仍由 Step 1 在分配前把 10210/10213 解析为真正的 actual kid；raw
C++ 直接调用若绕过 Python，则由 exact-N guard 拒绝不合法的 bf16 workspace shape。

### 5. `csrc/opus_gemm/codegen/gen_instances_gfx1250.py`

状态：修改，`+47/-62`。

修改位置：`splitk_reduce_extra_device_instantiations()` 和
`gen_cluster_tdm_splitk_ws_instance()`（提交后第 50-390 行）。

- gfx1250 extra reduce instantiation 和 launcher 内 forward declaration 全部改为
  `const void*`；
- split-K launcher 增加 `aiter_tensor_t &workspace`；
- C++ raw 最后防线新增真实的 `batch == 1` 检查；
- 保留原 split-K clamp 和整 tile/cluster 物理约束；
- clamp 后 checked 计算 `padded_M`、`padded_N`、slice numel 和 one-batch
  `[split_k, padded_M, padded_N]` required numel；
- 共享 validator 要求 fp32、同设备、contiguous、容量足够且 16-byte aligned；
- direct pointer 写入 `kargs.ptr_ws` 并传给所有 reduce launch；
- 保留 `Y=bf16 + bias=fp32` 的专门 reduce 分支，以及其他 bf16/fp32、bias/no-bias
  数值路径；
- 手写 host explicit instantiation 同步增加 workspace 参数。

删除 capture 检测、registry、raw allocation/grow、device handle mirror 和 mirror sync。

### 6. `csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh`

状态：修改，`+138/-185`。

#### 两套 function pointer 类型和 entry（提交后第 21-96 行）

- guarded 定义 5 参数 `OpusA16W16Kernel`；
- guarded 定义 6 参数 `OpusA16W16WorkspaceKernel`；
- runtime shape entry 改为 `{shape, int kid}`；
- non-workspace tune entry 和 workspace tune entry 使用不同 function pointer 类型；
- `workspace_entry()` 只构造 gfx950 generated workspace table。

#### strict dispatch（提交后第 117-164 行）

- `opus_a16w16_tune_dispatch_gfx950<bf16_t/fp32_t>()` 只查询当前架构、当前 dtype
  的 non-workspace 表；
- 新增 `opus_a16w16_workspace_dispatch_gfx950()`，返回
  `OpusA16W16WorkspaceKernel`；
- 新增 `opus_a16w16_has_workspace_kernel_gfx950()`，capability 完全来自 generated
  workspace table membership。

#### legacy selector probe（提交后第 166-203 行）

- `(M,N,K)` tuned lookup 命中后只返回 integer kid；
- miss 时仍调用原 gfx950 integer heuristic；
- bf16/fp32 的 4 GiB fallback guard 继续保留；
- 删除在 selector 内重新 dispatch 并 launch function pointer 的路径；
- 删除 `200..299` 及 nooob offset 的 split-K 数值区间副本。

### 7. `csrc/opus_gemm/include/gfx942/opus_gemm_arch_gfx942.cuh`

状态：修改，`+197/-99`。

- 与 gfx950 相同，新增严格分离的 5 参数/6 参数 function pointer、runtime kid entry、
  non-workspace entry 和 workspace entry；
- non-workspace tune dispatch 只查 gfx942 generated non-workspace table；
- 新增 gfx942 workspace membership probe 和 6 参数 workspace dispatch；
- runtime tuned lookup 只返回 kid。

原 gfx942 heuristic header 返回混合 5 参数 launcher function pointer，其中 split-K
launcher 在本步已经成为 6 参数，不能继续作为 selector 返回类型。因此本 arch header 内
新增等价的纯 integer heuristic：

- `split_barrier_ok()`、`bf16ws_band()`；
- `heuristic_bf16_kid()`；
- `heuristic_non_bf16_or_bias_kid()`。

分支顺序、shape 条件和返回 kid 与旧 heuristic 保持一致，返回 kid 均属于现有
`HEURISTIC_DEFAULT_KIDS_GFX942`。旧
`opus_gemm_heuristic_dispatch_gfx942.cuh` 文件本身没有删除或修改，继续保留作 parity
golden。gfx942 a8w8 的独立 5 参数 scale ABI 和 tune dispatch 保持不变。

### 8. `csrc/opus_gemm/include/gfx1250/opus_gemm_arch_gfx1250.cuh`

状态：修改，`+163/-99`。

- 不再借用 gfx950 detail entry/type，定义自己的 runtime、non-workspace 和 workspace
  entry；
- gfx1250 当前 non-workspace 表为空，但 bf16/fp32 strict dispatch 仍查询 generated
  size-0 `std::array`，错误 ABI 会在边界明确失败；
- 新增 `opus_a16w16_workspace_dispatch_gfx1250()` 和 generated table membership
  probe；
- 所有 496 个 canonical gfx1250 split-K kid 都只进入 6 参数 workspace 表；
- `(M,N,K)` tuned lookup 和 heuristic fallback 只返回 integer kid；
- 4 GiB descriptor guard 和原 gfx1250 integer heuristic 保留；
- 不再假设“gfx1250 所有 kid 都可通过 fp32 5 参数 tune table launch”。

### Step 3 验证结果

#### Python 和现有单测

```text
python -m py_compile \
  csrc/opus_gemm/gen_instances.py \
  csrc/opus_gemm/codegen/gen_instances_gfx950.py \
  csrc/opus_gemm/codegen/gen_instances_gfx942.py \
  csrc/opus_gemm/codegen/gen_instances_gfx1250.py

pytest -q op_tests/test_opus_dispatch.py op_tests/test_opus_workspace.py
80 passed, 2 warnings

git diff --check
通过
```

两条 warning 来自环境中 Cython `dep_util` deprecation，与本提交无关。

#### fresh codegen / generated ABI 检查

代表性生成目录：`/tmp/opus-step3-final.QlTh4j`。代表 kid 覆盖：

```text
gfx950: 200 workspace, 300 non-workspace
gfx942: 10000 non-workspace, 10200/10204 fp32 workspace,
        10210/10213/10216 bf16 workspace
gfx1250: 20000 workspace
```

检查结果：

- manifest 对 workspace launcher 生成 6 参数，对 non-workspace launcher 生成 5 参数；
- runtime shape lookup 六张 arch/dtype 表都只含 integer kid，不含 launcher address；
- generated split-K launcher 不含 `opus_splitk_ws_*`、`ws_handle`、`hipMalloc`、
  `hipFree`、`hipStreamIsCapturing` 或 `hipDeviceSynchronize`；
- 三个 reduce host/device generated TU 使用 `const void*`；
- 10210、10213、10216 都生成完整 exact-N guard，且 10210/10213 launcher 内不再引用
  10200/10203 redirect symbol；
- canonical a16w16 表分区与 registry 独立对拍通过：

| arch | non-workspace bf16 | non-workspace fp32 | workspace |
|---|---:|---:|---:|
| gfx950 | 92 | 92 | 48 |
| gfx942 | 14 | 1 | 8 |
| gfx1250 | 0 | 0 | 496 |
| 合计 workspace |  |  | 552 |

`SPLITK_TAGS` 生成分类与 canonical `SPLITK_KIDS` registry 的 552 个成员完全一致。

#### subset-compile 验证

用真实 CLI、无 tuned CSV、显式 `GPU_ARCHS` 分别生成三次：

| arch | sidecar compile set | heuristic defaults | workspace table members |
|---|---:|---:|---:|
| gfx950 | 10 | 8 | 6 |
| gfx942 | 15 | 15 | 7 |
| gfx1250 | 6 | 6 | 6 |

三次均确认：

- 当前架构的 heuristic defaults 全部留在 sidecar；
- sidecar 没有混入错误架构 kid；
- workspace table size 与 subset 中实际 workspace kid 一致；
- 其他架构生成 size-0 table，不引用未编译 launcher。

#### HIP header/type 检查

CK 子模块已初始化到 pinned commit
`f33252cebe5a52362ec1ee12c124dde7800dda3a`。使用 fresh generated manifest/lookup：

- gfx950 arch header 单独 `hipcc -fsyntax-only`：通过；
- gfx942 arch header 单独检查：通过；
- gfx1250 arch header 单独检查：通过；
- 三个 arch header 同一 TU 组合检查：通过。

只有仓库原有的 `aiter_tensor.h` 未消费 `hipFree*` 返回值和 `opus.hpp` deduction-guide
attribute warning，没有 Step 3 新错误。

#### gfx942 heuristic parity

把新 C++ pure-kid heuristic 编译成 host probe，与 Step 1 的 Python parity port 对拍
8712 组 `(M,N,K,dtype/bias mode)`，全部一致。corpus 覆盖各分支的边界值，包括
small-M WKC、bf16ws band、N=384、large-N、split-barrier、K=4096 特化和 fallback。

### Step 3 明确没有修改或删除的相关文件

| 文件/代码 | Step 3 状态 |
|---|---|
| 三架构 split-K traits/kargs | 尚保留 `ws_handle`；Step 4 才改 `void* ptr_ws` |
| 三架构 main pipeline | 尚未改 direct pointer 解引用；Step 4 才修改 |
| 三架构 reduce kernel definition | 尚未改 `const void*` definition；Step 4 才修改 |
| `csrc/opus_gemm/opus_gemm.cu` | 未修改；旧 arch router/raw tune entry/allocator registry 仍在，Step 5 才接通 |
| raw header、rocm ops、pybind | 未修改；Step 5 才增加 optional workspace |
| `aiter/ops/opus/gemm_op_a16w16.py` | 未修改；Step 2 准备路径仍未切到生产 raw binding |
| `opus_gemm_workspace_init()` 和 prewarm | 未删除；Step 5 才改 deprecated no-op并清理调用 |
| `opus_gemm_heuristic_dispatch_gfx*.cuh` | 文件均保留，继续作为 parity golden |
| `op_tests/test_opus_dispatch.py` / `test_opus_workspace.py` | 本步未修改，只作为回归测试运行 |

没有进行 GPU 数值、graph capture 或性能测试：Step 3 与后续 Step 4/5 共同组成 ABI
迁移，当前 traits/reduce definition/raw entry 尚未连接，新 generated launcher 不能作为完整
JIT 模块单独验收。下一步必须继续 Step 4，不能为临时编译恢复 handle shim 或混合函数指针
表。

## Step 4：把三架构 kernel 改成 direct pointer

- 日期：`2026-08-10`
- 分支：`splitk_to_torch_2`
- 修改前 HEAD：`0a9bd8101c1d8ac84d6734deaf2ec385a45c0e54`
- 完成提交：`34b70a8430273e5458862724fc781102d6fe5afe`
- 提交标题：`[OPUS] Use direct split-K workspace pointers`
- 合计：16 个代码文件，新增 95 行，删除 113 行
- ABI 状态：Step 3 generated launcher、Step 4 kargs/main/reduce 已统一为 direct pointer；
  `opus_gemm.cu` raw entry/registry 仍保留旧实现，按计划由 Step 5 接通和删除

### Step 4 文件总表

| 状态 | 文件 | 行数变化 |
|---|---|---:|
| 修改 | `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh` | `+7/-18` |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_traits_a16w16.cuh` | `+18/-14` |
| 修改 | `csrc/opus_gemm/include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh` | `+11/-27` |
| 修改 | `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh` | `+2/-3` |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_em3en4_lds1_pgr2_sk.cuh` | `+3/-3` |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_kbuf1.cuh` | `+2/-1` |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_kbuf2v.cuh` | `+2/-1` |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_kbuf2v_bk128.cuh` | `+3/-1` |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_quad_mfma32_kbuf1.cuh` | `+10/-8` |
| 修改 | `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh` | `+4/-3` |
| 修改 | `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh` | `+4/-3` |
| 修改 | `csrc/opus_gemm/include/gfx950/splitk_reduce_gfx950.cuh` | `+5/-6` |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/splitk_reduce_gfx942.cuh` | `+9/-9` |
| 修改 | `csrc/opus_gemm/include/gfx1250/splitk_reduce_gfx1250.cuh` | `+8/-8` |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_helpers_a16w16.cuh` | `+6/-6` |
| 修改 | `csrc/opus_gemm/include/opus_gemm_common.cuh` | `+1/-2` |

最初 Step 4 清单之外实际还修改了两个文件：

- `opus_gemm_helpers_a16w16.cuh` 中的 split-K workspace store helper 必须从旧语义名
  `D_C` 改为 `D_WS`，否则 workspace 转换类型仍没有完全收口；
- `opus_gemm_common.cuh` 只删除已经失效的“gfx950/gfx1250 共享 handle”注释，不增加
  arch/family/kid 逻辑。

### 1. `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh`

状态：修改，`+7/-18`。

- `opus_flatmm_splitk_traits_gfx950` 的 tuple slot 2 从 `D_C` 明确命名为 `D_WS`；
- fp32 workspace 的 `static_assert` 改为检查 `D_WS`；
- 删除本文件的 `opus_splitk_ws_handle`、
  `OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED` guard 及 graph-grow/device-mirror 注释；
- `opus_gemm_flatmm_splitk_kargs_gfx950` 的 `ws_handle` 字段替换为
  `void* ptr_ws`；
- workspace layout 仍为 `[split_k, batch, padded_M, padded_N]`，字段顺序及其他 stride
  不变。

### 2. `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_traits_a16w16.cuh`

状态：修改，`+18/-14`。

- 在通用 traits 的 tuple slot 2 上增加显式 `D_WS` alias；`D_C` 继续供 non-split
  output 分支使用；
- 删除 handle struct、guard 和旧 `opus_splitk_ws_ptr()`；
- 新增 `opus_gfx942_uniform_ws_ptr<D_WS>(ptr_ws)`：
  - 接受 `void*` 或 `const void*`，由入参 constness 推导返回 `D_WS*` 或
    `const D_WS*`；
  - 静态检查入参确为 cv-void pointer；
  - gfx942 device 路径仍由 lane 0 取得 64 位地址，拆成高低两个 32 位值，并执行两次
    `__builtin_amdgcn_readfirstlane` 后重组；
  - helper 不读取 handle、host slot 或 device mirror；
- `opus_gemm_splitk_kargs::ws_handle` 替换为 `void* ptr_ws`，其余 ABI 字段和 workspace
  layout 不变。

### 3. `csrc/opus_gemm/include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh`

状态：修改，`+11/-27`。

- 删除本文件的 handle struct 和 guard；
- `opus_gemm_cluster_tdm_ws_kargs_gfx1250` 改为 caller-owned `void* ptr_ws`；
- traits 模板参数和 alias 从 workspace 含义的 `D_C/DataC` 改为 `D_WS/DataWS`；
- fp32 workspace `static_assert` 改为检查 `D_WS`；
- workspace store vector `kCVec` 改为按 `sizeof(DataWS)` 计算，不再从 accumulation
  type 推导；
- layout 仍为 `[split_k, padded_M, padded_N]`，gfx1250 batch==1 约束未改变。

### 4. `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_helpers_a16w16.cuh`

状态：修改，`+6/-6`。

只修改 `epilogue_store_workspace_sc0nt()`：

- workspace store 的类型 alias 从 `D_C` 改为 `D_WS`；
- `D_WS == D_ACC` 快路径判断以及 fallback cast 全部改用 `D_WS`；
- non-split 的 `epilogue_store_c_if()` 和 `epilogue_store_c_lds_staged()` 继续使用
  `D_C`，未混淆 output 与 workspace 类型；
- sc0+nt cache policy、offset 公式、store vector 和 store 次序均未改变。

### 5. `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh`

状态：修改，`+2/-3`。

- kernel 局部 workspace type 改为 `T::D_WS`；
- 删除 `kargs.ws_handle->ptr` 解引用，直接执行
  `reinterpret_cast<D_WS*>(kargs.ptr_ws)`；
- split、batch、row、column 四段 workspace offset 公式保持原样。

### 6. `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_em3en4_lds1_pgr2_sk.cuh`

状态：修改，`+3/-3`。

- split-only pipeline 的 `D_C` workspace alias 改为 `D_WS`；
- fp32 workspace assertion 改为 `D_WS == D_ACC`；
- workspace base 改为
  `opus_gfx942_uniform_ws_ptr<D_WS>(kargs.ptr_ws)`；
- A/B swap、split/batch/row/column offsets 和 main-loop 行为不变。

### 7. gfx942 kbuf 双用途 pipelines

涉及：

- `opus_gemm_pipeline_a16w16_kbuf1.cuh`，`+2/-1`；
- `opus_gemm_pipeline_a16w16_kbuf2v.cuh`，`+2/-1`；
- `opus_gemm_pipeline_a16w16_kbuf2v_bk128.cuh`，`+3/-1`。

共同修改：

- 保留 `D_C` 给 non-split `ptr_c` 分支，新增 `D_WS` 给 split-K 分支；
- split-K workspace pointer 改为
  `opus_gfx942_uniform_ws_ptr<D_WS>(kargs.ptr_ws)`；
- non-split pointer、byte extent 和 stride 继续使用 `D_C/ptr_c`；
- `bk64_traits_view` 额外转发 `D_WS`，确保 B_K=128 pipeline 的内部 B_K=64 view 不
  丢失 workspace 类型；
- 三条 split-K store 路径继续调用已改为 `D_WS` 的
  `epilogue_store_workspace_sc0nt()`。

### 8. `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_quad_mfma32_kbuf1.cuh`

状态：修改，`+10/-8`。

- 保留 non-split `D_C`，新增 `D_WS`；
- 新增 `D_STORE = std::conditional_t<IS_SPLITK, D_WS, D_C>`，让共用 epilogue 在
  split/non-split 两种实例中取得正确的物理 store 类型；
- split pointer 改为 direct uniform helper，non-split pointer 仍为 `ptr_c`；
- C-stage LDS byte extent、bf16/fp32 cast、vector traits、最大 store vector、LDS view 和
  store chunk 全部改由 `D_STORE` 表达；
- quadrant layout、LDS stride、address offset 和 store 顺序不变。

### 9. gfx1250 两个 main pipeline

涉及：

- `opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh`，`+4/-3`；
- `opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh`，`+4/-3`。

共同修改：

- 新增 `DataWS = T::DataWS`；
- 删除 `kargs.ws_handle->ptr`，直接
  `reinterpret_cast<DataWS*>(kargs.ptr_ws)`；
- workspace byte extent 和 `make_gmem` element type 改为 `DataWS`；
- WMMA accumulation 继续使用 `DataAcc`；
- split/tile offset、cluster barrier、OOB tile 处理和 store 次序不变。

### 10. `csrc/opus_gemm/include/gfx950/splitk_reduce_gfx950.cuh`

状态：修改，`+5/-6`。

- reduce kernel 首参从 handle pointer 改为 `const void* ws_ptr`；
- 删除仅为 handle 定义引入的 traits header；
- 本架构固定 fp32 workspace，用局部 `using D_WS = float` 后直接 cast 为
  `const D_WS*`；
- workspace buffer byte extent 改为 `sizeof(D_WS)`；
- reduction、bias、tail/OOB 和 output cast/store 逻辑不变。

### 11. `csrc/opus_gemm/include/gfx942/a16w16/splitk_reduce_gfx942.cuh`

状态：修改，`+9/-9`。

以下定义和所有转发层均统一为 `const void* ws_ptr`：

- `splitk_reduce_kernel_fallback_body()`；
- `splitk_reduce_kernel_fallback()`；
- `splitk_reduce_kernel_bf16ws_fallback()`；
- `splitk_reduce_kernel_exact_n_rowblock()`。

baseline fp32ws、bf16ws fallback、exact-N row-block 及 HAS_OOB template 路径都通过
`opus_gfx942_uniform_ws_ptr<D_WS>(ws_ptr)` 取得 const direct pointer；wrapper 只转发
`ws_ptr`，不再出现 handle 或旧 helper。

### 12. `csrc/opus_gemm/include/gfx1250/splitk_reduce_gfx1250.cuh`

状态：修改，`+8/-8`。

- 首参改为 `const void* ws_ptr`，删除 handle-only traits include；
- 用局部 `D_WS=float` 直接 cast workspace，并用 `sizeof(D_WS)` 计算 byte extent；
- non-gfx1250 stub 的 unused argument 同步改为 `ws_ptr`；
- kernel 名字继续与 gfx950 区分，reduce/bias/OOB 数值路径不变。

### 13. `csrc/opus_gemm/include/opus_gemm_common.cuh`

状态：修改，`+1/-2`。

仅更新 gfx1250 traits include 上方的注释：删除“与 gfx950 共享 guarded handle”描述，
改为 direct-pointer kargs。Step 3 加入的 family-neutral checked extent 和 workspace validator
没有修改，也没有在 common 层加入 a16w16/arch/kid 特判。

### Step 4 验证结果

#### Python 回归测试

```text
pytest -q op_tests/test_opus_dispatch.py op_tests/test_opus_workspace.py
80 passed, 2 warnings

git diff --check
通过
```

两条 pytest warning 仍来自环境中的 Cython `dep_util` deprecation。

#### fresh representative codegen 和 HIP 编译

fresh 生成目录：`/tmp/opus-step4-codegen.yechor`。代表 kid 同时覆盖改动过的 gfx942
双用途 pipeline 的 non-workspace 分支：

```text
gfx950: 200
gfx942 non-workspace: 10000, 10001, 10003, 10006
gfx942 workspace: 10200, 10204, 10210, 10213, 10216
gfx1250: 20000 (plain cluster/TDM), 20100 (cluster-launch/TDM)
```

结果：

- 三架构 `all_instances_host_<arch>.cu` 均通过对应 target 的
  `hipcc -fsyntax-only`；
- 13 个 main device TU 均通过（kid 10000 同时生成 bf16/fp32 output specialization）；
- 三个 `splitk_reduce_<arch>.device.cu` 均通过；
- generated host launcher 的 `kargs.ptr_ws = workspace_ptr_` 与 Step 4 traits 匹配；
- generated main/reduce 调用均传 direct pointer，fresh 产物中不存在
  `opus_splitk_ws_*`、`ws_handle`、`hipMalloc`、`hipFree`、capture query 或 device sync。

编译只出现仓库/环境已有 warning：`--hip-link` unused、`aiter_tensor.h` 未消费
`hipFree*` 返回值、`opus.hpp` deduction-guide attribute，以及 gfx942 helper 的 duplicate
`inline`；没有 Step 4 编译错误。

#### header 组合检查

将 `opus_gemm_common.cuh`、gfx942 traits 和三架构 reduce headers 组合在同一 HIP TU，
分别以 gfx950、gfx942、gfx1250 为 target 执行 `-fsyntax-only`，三次均通过。删除三份
guarded handle 后没有跨架构重复定义或签名冲突。

#### gfx942 direct-pointer uniform ISA 检查

对 fresh gfx942 main TU 和 reduce TU 执行 `--cuda-device-only -S`：

- main kernel 汇编中 direct workspace 地址重组仍出现相邻两条
  `v_readfirstlane_b32`，对应 64 位地址的低/高 32 位；
- reduce 的 baseline/bf16ws/exact-N 显式实例同样保留成对
  `v_readfirstlane_b32`；
- 因此本步删除的是 handle/device-mirror 读取，没有删除既有 wave-uniform pointer
  语义。

#### 旧符号和 workspace 类型扫描

- `csrc/opus_gemm/include` 内以下符号全部清零：
  `opus_splitk_ws_handle`、`OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED`、
  `opus_splitk_ws_ptr`、`kargs.ws_handle`、`ws_handle`；
- workspace pointer cast、workspace byte extent、workspace store conversion 均来自
  `D_WS/DataWS/D_STORE`；
- 全 `csrc/opus_gemm` 扫描仍只在 `opus_gemm.cu` 的旧 registry 和 README 说明中命中
  handle；这是 Step 5 的明确删除范围，不在 Step 4 恢复兼容 shim。

### Step 4 明确没有修改或删除的相关文件

| 文件/代码 | Step 4 状态 |
|---|---|
| `csrc/opus_gemm/opus_gemm.cu` | 未修改；旧 allocator/registry/raw router 仍在，Step 5 删除并接 direct workspace raw entry |
| `csrc/opus_gemm/include/opus_gemm.h` | 未修改；Step 5 才增加 raw optional workspace 参数 |
| `csrc/include/rocm_ops.hpp`、`csrc/pybind/opus_gemm_pybind.cu` | 未修改；Step 5 才同步 binding |
| `aiter/ops/opus/gemm_op_a16w16.py` | 未修改；Step 2 准备路径尚未切为生产 raw 调用 |
| `aiter/ops/opus/common.py`、`aiter/tuned_gemm.py` | 未修改；prewarm/init 清理由 Step 5 完成 |
| 三架构 codegen/dispatch | 未修改；继续使用 Step 3 已生成的 5/6 参数分表和 `const void*` ABI |
| main/reduce workspace layout 和地址公式 | 未改变；只替换 pointer 来源和显式 dtype 名称 |

本步没有运行 GPU 数值、graph capture/replay 或性能测试。原因是 raw entry 与 Python
生产入口必须到 Step 5 才完整接通；Step 4 验收的是 generated launcher、kernel kargs、
main pipeline 和 reduce definition 的内部 ABI 一致性。当前不能为编译旧
`opus_gemm.cu` 恢复 handle shim，下一步必须直接完成 Step 5。

## Step 5：接通 raw entry/binding 并删除旧 allocator

完成时间：2026-08-10。

独立提交：

```text
4e8ce216eed77a94fbb504cc26585ce6afef8b5f
[OPUS] Route split-K through Torch workspaces
```

提交相对 Step 4 `34b70a8430273e5458862724fc781102d6fe5afe` 的实际统计：

```text
7 files changed, 134 insertions(+), 409 deletions(-)
```

### Step 5 文件总表

| 状态 | 文件 | numstat |
|---|---|---:|
| 修改 | `csrc/opus_gemm/include/opus_gemm.h` | `+1/-3` |
| 修改 | `csrc/include/rocm_ops.hpp` | `+4/-11` |
| 修改 | `csrc/pybind/opus_gemm_pybind.cu` | `+0/-1` |
| 修改 | `csrc/opus_gemm/opus_gemm.cu` | `+76/-252` |
| 修改 | `aiter/ops/opus/gemm_op_a16w16.py` | `+47/-45` |
| 修改 | `aiter/ops/opus/__init__.py` | `+6/-1` |
| 修改 | `aiter/tuned_gemm.py` | `+0/-96` |

没有修改三个保持公共 wrapper 调用方式的调用方：

```text
csrc/opus_gemm/opus_gemm_tune.py
csrc/gemm_a16w16/gemm_a16w16_tune.py
aiter/ops/deepgemm.py
```

它们继续把 `(bias, kernelId, splitK)` 或旧 positional `(kernelId, splitK)` 传给 Python
公共 wrapper；workspace 规划和分配没有复制到调用方。

### 1. `csrc/opus_gemm/include/opus_gemm.h`

- `opus_gemm_a16w16_tune()` 在 `bias` 与 `kernelId` 之间新增无默认值的
  `std::optional<aiter_tensor_t> workspace`；raw C++ ABI 现在固定为 7 个参数；
- 删除 `opus_gemm_workspace_init()` C++ 声明；
- a8w8 blockscale tune entry 完全未改。

### 2. `csrc/include/rocm_ops.hpp`

- `OPUS_GEMM_A16W16_TUNE_PYBIND` 同步为
  `XQ, WQ, Y, bias, workspace, kernelId, splitK`；
- raw binding 的 `bias/workspace/kernelId/splitK` 都不提供 pybind 默认值，兼容默认值只在
  Python 公共 wrapper 上维护；
- 删除 `OPUS_GEMM_WORKSPACE_INIT_PYBIND`；
- a8w8 binding macro 未改。

### 3. `csrc/pybind/opus_gemm_pybind.cu`

- 删除 `OPUS_GEMM_WORKSPACE_INIT_PYBIND` 注册调用；
- `AITER_SET_STREAM_PYBIND`、generic opus、a16w16 tune 与 a8w8 tune 注册保持不变；
- 重建后的 `module_deepgemm_opus` 不再导出 C++
  `opus_gemm_workspace_init` 符号。

### 4. `csrc/opus_gemm/opus_gemm.cu`

#### raw tune 的两套 ABI 路由

- 删除旧 `opus_kid_is_splitk()` 及 gfx950/gfx942/gfx1250 workspace 数值区间副本；
- 新增当前架构 router：
  - `opus_a16w16_has_workspace_kernel()` 只查询 Step 3 generated workspace table；
  - `opus_a16w16_workspace_dispatch()` 只返回当前架构的
    `OpusA16W16WorkspaceKernel`；
- workspace table 命中时：
  - raw entry 要求 `workspace.has_value()`；
  - 要求 Y 为 bf16/fp32；
  - 调用六参数 launcher
    `(XQ, WQ, Y, workspace.value(), bias, splitK)`；
- workspace table 未命中时：
  - raw entry 要求 `workspace == std::nullopt`；
  - 按 Y dtype 查询原 bf16/fp32 non-workspace 表；
  - 调用五参数 launcher `(XQ, WQ, Y, bias, splitK)`；
- 错架构 kid 不会因共享数值范围命中；是否属于 workspace ABI 的唯一来源是当前架构
  generated table membership。

#### C++ workspace 最后防线

raw entry 检查 workspace 是否存在/是否多传；命中后的 generated launcher 继续调用 Step 3
加入的 `opus_validate_workspace()`，按 actual kid 的 `D_WS` 检查：

- `workspace.device_id == XQ.device_id`；
- dtype 精确等于 fp32 或 gfx942 kid 指定的 bf16；
- contiguous；
- non-null pointer 与 16-byte alignment；
- clamp 后的 `required_numel <= workspace.numel()`；
- `padded_M`、`padded_N`、slice、split、batch 和 byte span 的每段乘法均先经过
  `opus_checked_extent_product()` 的 `size_t` overflow 检查。

因此不存在只按 numel 猜 dtype/byte width 的路径；raw entry 也没有重新复制
a16w16 tile/arch/kid 规划逻辑。

#### generic bf16 停用与 allocator 删除

- `opus_gemm()` 的 fp8/a8w8 分支逐行保留；
- generic bf16 分支改为明确报错，要求调用 Python `gemm_a16w16_opus` 或
  `opus_gemm_a16w16_tune`，避免绕过 actual-kid/workspace planner；
- 删除不再可用的 generic a16w16 direct-launch router；
- 整体删除：
  - `SplitkWsRegistry`；
  - `opus_splitk_ws_get()`；
  - `opus_splitk_ws_device_handle()`；
  - `opus_splitk_ws_sync_to_device()`；
  - `opus_gemm_workspace_init()` C++ 实现；
  - host-pinned handle、device mirror、`hipMalloc`/`hipHostMalloc`/`hipMemcpy`、capture
    query 和同步逻辑；
- 删除只为 registry 使用的 `<mutex>`、`<unordered_map>` include；没有恢复 handle shim。

### 5. `aiter/ops/opus/gemm_op_a16w16.py`

#### raw binding 与公共兼容 ABI

- raw fake/schema 与 `_opus_gemm_a16w16_tune_raw()` 同步为 7 个必传参数，其中
  `workspace` 为 `Tensor | None`；
- 公共 `opus_gemm_a16w16_tune()` 保持旧参数顺序
  `(XQ, WQ, Y, bias=None, kernelId=0, splitK=0)`，只在末尾新增 keyword-only
  `workspace=None`；
- 旧 positional `(..., kid, splitK)` 迁移逻辑继续生效；
- 显式 workspace 只能通过 keyword 传入，不会把旧 positional kid 解释成 Tensor。

#### 集中 Torch workspace 路径正式启用

- `_launch_a16w16_with_torch_workspace()` 现在是生产路径，并集中执行：
  1. XQ/WQ/Y layout 验证；
  2. 从已经解析的 `LaunchConfig.actual_kid` 取得 canonical instance；
  3. `plan_a16w16_workspace()`；
  4. 缺失且需要 workspace 时调用 `allocate_workspace()`/`torch.empty`；
  5. 显式 Tensor 时调用同一个 `validate_workspace()`，不分配；
  6. raw launch 传 `workspace | None, actual_kid, launch_split_k`；
- `opus_gemm_a16w16_tune()` 先解析 requested/actual kid，再进入上述路径；
- `gemm_a16w16_opus()` 复用第一次 selector 已得到的完整 config，直接进入上述路径，
  不进行第二次 selector；
- non-workspace plan 返回 `None`，若用户显式传 workspace 则在 Python 边界拒绝；
- 没有 module/global/per-shape workspace Tensor cache，每次缺失 workspace 都创建新的调用级
  Tensor。

#### deprecated init

- 删除 `@compile_ops(... fc_name="opus_gemm_workspace_init")`；
- `opus_gemm_workspace_init()` 变为纯 Python no-op，只发出 `DeprecationWarning`；
- 调用该函数不会注册 stream、分配显存或触发 C++ workspace init。

### 6. `aiter/ops/opus/__init__.py`

- 支持架构继续从 `gemm_op_a16w16.py` 导出 deprecated Python no-op；
- 不支持架构不再把该名字设为 arch error stub，而是惰性转发到同一个 no-op；
- 因此旧代码在任何架构上调用 init 都只收到 deprecation warning，不再因架构或 capture
  状态失败。

### 7. `aiter/tuned_gemm.py`

删除全部旧 raw allocator prewarm 协议，共 `-96` 行：

- `_opus_is_splitk_kid` / `_opus_workspace_init` import；
- `_OPUS_WS_ARCHS`；
- `_opus_ws_warmed_sigs` 全局 set；
- `_opus_needs_ws_prewarm()`；
- `_opus_graph_capture_stream()` 默认 capture-stream 猜测；
- `_opus_prewarm_capture_workspace()`；
- `opus_gemm()` 内 eager prewarm 调用点和说明。

正常 `_opus_tune(...)` 调用保持原样，由公共 wrapper 为当前调用自动持有 workspace Tensor；
本文件不再创建额外 stream、不再同步 stream，也不缓存 shape signature。

### Step 5 验证结果

#### Python 静态检查与公共 wrapper 行为

```text
python -m py_compile \
  aiter/ops/opus/gemm_op_a16w16.py \
  aiter/ops/opus/__init__.py \
  aiter/tuned_gemm.py

python -m compileall -q aiter/ops/opus aiter/tuned_gemm.py
git diff --check
```

全部通过。

独立 CPU/fake-raw 行为探针通过：

- workspace kid 200 自动分配 shape `(2, 1, 128, 64)`、dtype fp32 的 Tensor，并传给
  raw binding；
- keyword-only 显式 Tensor 原对象复用，不调用 allocator；
- non-workspace kid 300 向 raw binding 传 `None`；
- 旧 positional `(kid, splitK)` 仍被解释为 `(200, 2)`；
- deprecated init 返回 `None`，只产生一条 `DeprecationWarning`。

#### 现有 CPU 测试与 Step 6 待更新断言

不含两个 pre-Step-5 行为断言的结果：

```text
pytest -q op_tests/test_opus_dispatch.py op_tests/test_opus_workspace.py \
  -k 'not test_csv_miss_production_path_uses_tune_wrapper_not_generic_cpp \
      and not test_step2_production_path_does_not_enable_torch_workspace_yet'

78 passed, 2 deselected, 2 warnings
```

完整运行结果为 `78 passed, 2 failed, 2 warnings`；两项失败不是未预期的实现错误，而是
测试仍断言 Step 5 之前的调用边界：

1. `test_step2_production_path_does_not_enable_torch_workspace_yet` 明确要求不得调用
   `allocate_workspace()`；Step 5 的目标正是启用该调用；
2. `test_csv_miss_production_path_uses_tune_wrapper_not_generic_cpp` monkeypatch 公共 tune
   wrapper，但 Step 5 高层入口已按 Step 2 预备设计使用首次解析的 config 直接进入集中 raw
   helper；测试没有拦截新的 raw boundary。

这两项在 Step 6 改写为“workspace 自动分配/传递”和“generic C++ 不运行”的新断言；本提交
没有提前修改 Step 6 清单中的测试文件。两条 warning 仍来自环境中的 Cython
`dep_util` deprecation。

#### fresh 多架构 codegen 与 HIP 语法检查

fresh 目录：`/tmp/opus-step5-codegen.YgHY5P`。

```text
GPU_ARCHS='gfx942;gfx950;gfx1250' \
python csrc/opus_gemm/gen_instances.py \
  --working_path /tmp/opus-step5-codegen.YgHY5P
```

- subset 含 31 个 heuristic/必需 kid；
- generated `opus_build_archs.h` 同时启用 gfx942/gfx950/gfx1250；
- workspace table size 分别为 gfx950=6、gfx942=7、gfx1250=6；
- 使用该 fresh manifest/lookup/build-arch header，`opus_gemm.cu` 以 gfx950 target 执行
  `hipcc -std=c++20 -fsyntax-only` 通过；一次 host parse 同时覆盖三架构 router/header；
- `opus_gemm_pybind.cu` 使用同一 fresh headers 执行 `-fsyntax-only` 通过；
- 只有仓库既有的 `aiter_tensor.h` unused-result、`opus.hpp` deduction-guide attribute 与
  `--hip-link` unused warning，没有 Step 5 编译错误。

#### 真实 JIT/binding 验证

现有 gfx950 环境完成一次 `module_deepgemm_opus` 完整 JIT 重建和加载（约 14.5 秒），证明
raw C++、generated host/device TU、pybind 与链接 ABI 一致。重建后的 pybind 首行签名为：

```text
opus_gemm_a16w16_tune(
  XQ, WQ, Y, bias: aiter_tensor_t | None,
  workspace: aiter_tensor_t | None,
  kernelId, splitK) -> None
```

并确认：

- Python 公共签名仍是
  `(XQ, WQ, Y, bias=None, kernelId=0, splitK=0, *, workspace=None)`；
- `.so` 不再导出 `opus_gemm_workspace_init`；
- raw workspace kid 缺失 Tensor 明确报 `requires a workspace tensor`；
- raw non-workspace kid 多传 Tensor 明确报 `requires workspace=None`；
- generated launcher 的错误 dtype 负例在 launch 前由 C++
  `opus_validate_workspace()` 拒绝。

#### 负例探针说明

一次额外 raw capacity 负例最初把 gfx950 kid 200、K=256、requested split-K=2 与
8191-element workspace 组合。launcher 按既有物理约束把 effective split-K clamp 到 1，
因此实际 required_numel 是 4096，8191 并不短；该错误 fixture 使用 CPU Tensor 指针却到达
GPU kernel，触发一次 GPU memory fault。该结果不计入容量验证，生成的 151 MiB
`gpucore.103571.gpu` 已立即删除，未修改仓库数据。Step 6 的 runtime capacity 负例必须按
clamp 后 effective split-K 计算 exact/short-one 容量，并只使用真实 device Tensor；在此之前
不再执行可能到达 kernel 的 CPU raw probe。

#### 旧状态扫描

代码路径（README 留给 Step 6）中以下符号全部清零：

```text
SplitkWsRegistry
opus_splitk_ws_*
ws_handle
OPUS_GEMM_WORKSPACE_INIT_PYBIND
_opus_needs_ws_prewarm
_opus_prewarm_capture_workspace
_opus_workspace_init
_opus_is_splitk_kid
hipStreamIsCapturing / hipHostMalloc（OPUS workspace 路径）
```

旧 allocator/handle/prewarm 说明仍只存在于两份 OPUS README，按施工顺序留给 Step 6 文档
收尾；没有为通过编译恢复兼容 shim。

### Step 5 明确留给 Step 6 的工作

- 改写上述两项 pre-Step-5 CPU 测试并补 raw/公共 wrapper 的 missing/wrong
  dtype/device/noncontiguous/alignment/exact-capacity/short-one 覆盖；
- 三架构真实 GPU 数值、bf16/fp32 workspace、gfx942 redirect、gfx1250 batch/bias 回归；
- graph capture/replay、双 stream/TBO 并发与无全局 Tensor cache 生命周期验证；
- split-K 上限、scope isolation 与 a8w8 回归；
- 更新 `csrc/opus_gemm/README.md` 和 `aiter/ops/opus/README.md`，删除 registry/prewarm
  旧说明；
- Step 6 完成前不宣称 GPU 数值、graph 或性能验收已通过。

## Step 6：测试、ISA 对照与文档收尾

Step 6 已作为独立提交完成：

```text
b72e4cc414d843f592a4115a8dbd0da949dedada
[OPUS] Cover Torch workspace lifecycle
```

本提交只包含计划指定的 6 个 tracked 文件：

| 文件 | numstat | Step 6 作用 |
|---|---:|---|
| `op_tests/test_opus_dispatch.py` | `+231/-12` | 三架构 selector parity、gfx942 resolver/redirect/uniform pointer、scope isolation |
| `op_tests/test_opus_workspace.py` | `+251/-12` | 集中生产路径、raw C++ exact/short-one 和完整 workspace 负例 |
| `op_tests/test_opus_graph.py` | `+198/-0` | 新增 graph、双 stream、生命周期与 deprecated no-op 回归 |
| `op_tests/test_opus_a16w16_gemm.py` | `+127/-9` | 改成标准 pytest 收集并增加三架构条件数值/bias 用例 |
| `csrc/opus_gemm/README.md` | `+153/-271` | 重写 C++ dispatch、direct pointer、验证和 ISA 说明 |
| `aiter/ops/opus/README.md` | `+232/-995` | 重写用户 API、planner、显式复用、graph、bias 与排障说明 |

提交总计 `1192 insertions, 1299 deletions`。本任务的未跟踪 `docs/` 文件没有混入代码
提交。

### `op_tests/test_opus_dispatch.py`

Step 5 留下的旧生产边界断言已改写：CSV miss 现在 monkeypatch
`_opus_gemm_a16w16_tune_raw`，确认高层入口直接传递
`(XQ, WQ, Y, bias, workspace, actual_kid, launch_split_k)`，并继续把 generic C++ bf16
entry 设为 must-not-run。使用 `(M,N,K)=(65,33,512)` 验证实际 kid 200 和 fp32
`(1,1,128,64)` workspace。

新增/扩展覆盖：

- gfx950 对冻结的旧 C++ 分支函数做 M/N/K/bias 边界笛卡尔扫描；
- gfx1250 对冻结的旧 C++ 分支函数做 M/N 边界扫描；
- gfx942 既保留逐分支 expected-kid corpus，又把 7 个代表 workspace kid、4 种 requested
  split-K 与冻结 generated-launcher resolver 独立对照；
- gfx942 非 exact-N 集合取 `63/65/768/2049`，确认 `10210 -> 10200`、
  `10213 -> 10203` 和 `10216` 拒绝；
- 遍历完整 exact-N 集合 `{64,128,256,384,512,1024,2048}`，并保留独立 `N=384`
  用例；
- 静态检查 gfx942 traits helper 直接接收 `ptr_ws`、保留两次
  `__builtin_amdgcn_readfirstlane`，5 个 main pipeline 和 reduce 都调用 direct-pointer
  uniform helper，且不再包含 handle 符号；
- 快照 a8w8 tune、a8w4 stage1/stage2 公共 Python 参数，防止 a16w16 workspace 参数污染
  其他 family API。

该文件最终单独结果为 `118 passed`（随后纳入四文件总结果）。

### `op_tests/test_opus_workspace.py`

删除 `test_step2_production_path_does_not_enable_torch_workspace_yet` 的旧语义，替换为
生产入口自动分配并向 raw binding 传 workspace 的断言。新增两个连续自动调用持有不同
Tensor/地址，以及 split-K 超 K-tile 上限时 `allocate_workspace()` 调用次数仍为 0。

真实 raw C++ fixture 按架构定义为：

| arch | kid | `(M,N,K,splitK)` | exact workspace |
|---|---:|---|---|
| gfx950 | 200 | `(64,64,512,2)` | 8192 fp32 elements |
| gfx942 | 10200 | `(128,128,512,2)` | fp32 plan |
| gfx942 | 10210 | `(128,128,512,2)` | bf16 plan |
| gfx1250 | 20000 | `(16,32,512,2)` | fp32 plan |

所有可能到达 kernel 的 raw fixture 都使用真实 device Tensor。覆盖：

- exact capacity 实际 launch 成功；
- short-one 在 host 侧报 capacity error；
- missing、wrong dtype、noncontiguous、错 16-byte alignment 全部在 host 侧拒绝；
- 8-GPU 环境用输入 `cuda:0`、workspace `cuda:1` 验证 C++ device-id 防线；
- non-workspace gfx950 kid 300 多传 Tensor 要求 `workspace=None`；
- gfx1250 条件用例绕过 Python，直接确认 raw launcher 对 `batch=2` 的 C++ 最后防线。

这修正了 Step 5 探针曾忽略 clamp 的问题：gfx950 K=512、B_K=64、prefetch=4 时
requested split-K=2 不会被降到 1，因此 8192/8191 确实是 exact/short-one，CPU 指针不会
误入 GPU kernel。

### `op_tests/test_opus_graph.py`（新增）

新增文件包含三架构条件 fixture；本机实际执行 gfx950：

- 先用无效 kid 只加载 JIT/raw binding，不运行任何 workspace kernel；
- 首次目标 shape 调用发生在 `torch.cuda.graph` capture 内，记录恰好一次
  `torch.empty` workspace 分配；
- 3 组新 A/B 数据 replay，均与 fp32 torch golden 一致，replay 不再次进入 Python
  allocator；
- 把 deprecated init 替换为 must-not-run，证明 capture 路径不依赖 prewarm；
- 两个真实 CUDA stream 各自 launch，保留两个 workspace Tensor 至同步后，确认对象和
  `data_ptr()` 都不同，并分别与 golden 一致；
- 多个 M/N shape 的 fake-raw 调用只保存 workspace weakref，GC 后全部失效；扫描
  `gemm_op_a16w16`、`_workspace`、`_workspace_a16w16` 模块没有全局 Tensor；
- deprecated `opus_gemm_workspace_init()` 只返回 `None` 并产生
  `DeprecationWarning`。

### `op_tests/test_opus_a16w16_gemm.py`

删除 import-time `sys.exit(0)`，原 CLI smoke/sweep helper 改名为 `run_*`，因此文件可以被
常规 pytest 收集；不支持的架构只在具体 case 内 skip，CLI 的 skip 仍留在 `__main__`。

新增三架构条件矩阵：

- gfx950 kid 200：Y=bf16/fp32，split-K=2；
- gfx942 kid 10200（fp32 workspace、Y=fp32）和 kid 10210（bf16 workspace、Y=bf16）；
- gfx1250 kid 20000：Y=bf16/fp32，split-K=2。

bias 条件回归：

- gfx950 实机确认 Y=bf16 + bias=bf16 数值正确，bias=fp32 继续由 launcher 的
  match-Y dtype 规则拒绝；
- gfx942 保留自动 framework fallback，显式 workspace kid + bias 严格拒绝；
- gfx1250 条件用例覆盖 Y=bf16 + bias=fp32。

### 两份 README

两份 README 原内容同时包含“仅 gfx950”、generic shape selector 直接 launch、
`hipMallocAsync/hipFreeAsync`、stream registry/handle、capture 前 init/prewarm 等互相矛盾
说明。Step 6 将其改为当前实现：

- 三架构 actual-kid-first Python 选择；
- generated non-workspace/workspace 5/6 参数分表；
- typed `WorkspacePlan`、每调用 `torch.empty`、显式 Tensor 共享验证/复用；
- C++ checked extent + device/dtype/contiguous/alignment/capacity 双重防线；
- gfx942 exact-N redirect 和 direct pointer uniform helper；
- gfx1250 batch/bias 规则；
- graph capture 无 OPUS prewarm、双 stream 调用级所有权、无 Tensor cache；
- deprecated init 是 Python warning-only no-op；
- a8w8/a8w4 scope isolation 和新的 focused test 命令。

README 明确说明 gfx942 性能仍需 gfx942 硬件，未把交叉编译结果写成实机性能通过。

### 实际运行环境与 focused pytest

```text
torch                  2.11.0+rocm7.14.0
ROCm / HIP             7.14.60850
device_count           8
device                 AMD Instinct MI355X
gcnArchName            gfx950:sramecc+:xnack-
CU                     256
```

最终命令：

```text
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py

159 passed, 14 skipped, 2 warnings in 4.30s
```

14 个 skip 全部是当前机器缺少 gfx942/gfx1250 的架构条件 case；两条 warning 仍是环境中
Cython `dep_util` deprecation。当前 gfx950 实际通过范围包括数值、bias、raw C++ 全负例、
跨 device、graph 三次 replay、双 stream 和生命周期。gfx942/gfx1250 的 selector/planner
CPU 测试与条件测试已加入，但没有对应硬件数值/graph/bias/batch 实机结果。

### fresh 三架构 codegen / HIP syntax

fresh 目录：`/tmp/opus-step6-codegen.nFnIx1`。

```text
GPU_ARCHS='gfx942;gfx950;gfx1250' \
python csrc/opus_gemm/gen_instances.py \
  -w /tmp/opus-step6-codegen.nFnIx1
```

生成 31-kid 默认 subset。对目录内全部 40 个 `all_instances_host_<arch>.cu`、per-kid
device TU 和三架构 reduce TU，按文件所属架构执行 `hipcc -std=c++20 -O3
--offload-arch=<arch> -fsyntax-only`，40/40 通过。没有 Step 6 compile error。

### gfx942 迁移前后 ISA / register 对照

对照基线为 Step 3 提交 `0a9bd810`（direct-pointer 改动之前），当前为 Step 6 HEAD。
分别 fresh generate 后以完全相同 gfx942 device flags 编译 representative main kid 10200 与
完整 reduce TU：

```text
/tmp/opus-step6-before-gfx942-main.s
/tmp/opus-step6-after-gfx942-main.s
/tmp/opus-step6-before-gfx942-reduce.s
/tmp/opus-step6-after-gfx942-reduce.s
```

Step 3 的 reduce codegen declaration 已提前变为 `const void*`，而当时 definition 仍是
handle ABI，因此该中间提交的 generated reduce TU 本来不能单独实例化。仅在 `/tmp` 副本
中把 explicit-instantiation 第一参数机械还原为当时真实
`const opus_splitk_ws_handle*`，未修改仓库，再生成“before” ISA。

main kid 10200 结果：

| 指标 | before handle | after direct pointer |
|---|---:|---:|
| assembly lines | 848 | 838 |
| kernarg bytes | 96 | 96 |
| LDS bytes | 33792 | 33792 |
| next-free VGPR | 169 | 169 |
| next-free SGPR | 96 | 96 |
| `v_readfirstlane_b32` | 5 | 5 |
| handle `s_load_dwordx2 ... 0x0` | 1 | 0 |

完整 reduce TU 结果：

- before/after 都是 134 个 kernel；
- `v_readfirstlane_b32` 总数均为 276；
- 130 个 kernel SGPR 不变为 18，4 个 bias kernel 从 22 改善到 21；
- 按 gfx942 四-VGPR allocation block 取整后，124 个 kernel 不变，10 个减少一个 block，
  没有任何 kernel 增加 hardware VGPR allocation block。

因此 direct pointer 删除了 handle/device-mirror load，同时 main 与 baseline/bf16ws/exact-N
reduce 全部保留等价的 wave-uniform/readfirstlane 语义；静态 ISA/寄存器没有 occupancy
回退。当前机器没有 gfx942，不能完成 split-K latency/throughput 的迁移前后实测，性能项
明确保留为 gfx942 硬件 follow-up，未伪报通过。

### scope isolation

以任务基线 `ca68b4f3` 和当前 fresh codegen 比较：

- 两个 gfx950 a8w8/a8w8-scale generated launcher impl 逐字相同；
- 两个对应 device TU 逐字相同；
- `opus_gemm_a8w8_tune_lookup.h` 逐字相同；
- `csrc/opus_moe`、a8w4 stage1/stage2 Python、a8w8 Python 和四份 gfx950 a8w8
  pipeline/traits 从基线到当前无 tracked diff；
- 测试确认 a8w8/a8w4 公共 Python 参数列表未增加 a16w16 workspace。

### 最终机械检查

- `git diff --check`：通过；
- Python 测试文件 `py_compile`：通过；
- 代码路径（排除 README 的历史/扫描文字）中
  `SplitkWsRegistry|opus_splitk_ws_|ws_handle`：0 命中；
- allocator 路径中 `hipMalloc|hipFree|hipHostMalloc|hipStreamIsCapturing`：0 命中；
- `sys.path.insert/append`：0 命中；
- CK 仍为 pinned `f33252cebe5a52362ec1ee12c124dde7800dda3a`。

Step 1 至 Step 6 的代码实现、当前硬件可执行验收和文档收尾至此完成。唯一不能在本机
闭环的是 gfx942/gfx1250 实机数值/graph/bias/batch 以及 gfx942 性能；对应条件测试和
明确执行边界已保留，不能把 skip 或交叉编译写成硬件通过。

## 任务一最终完成定义收尾审计

用户在 Step 6 后补充了最终完成定义及四条必须原样执行的机械检查。本轮按该定义重新审计，
并以独立提交收尾：

```text
2352c46c784d6ba3a0c71ff89b4bdb4c2fefa59f
[OPUS] Finalize workspace migration audit

aiter/ops/opus/README.md |  7 +++++++
csrc/opus_gemm/README.md | 28 ++++++++++++----------------
2 files changed, 19 insertions(+), 16 deletions(-)
```

### 唯一发现及修复

第一次原样运行 legacy-symbol 扫描时，唯一命中来自 `csrc/opus_gemm/README.md`：README
自己抄录了包含 forbidden identifiers 的 `rg` 命令。因此虽然代码路径已经干净，扫描整个
目录仍会自命中。收尾提交删除了这段命令副本，改为引用任务完成清单，并明确 README 也不能
保存会被该扫描捕获的 literal identifier。

两份 README 同时补充以下架构边界：

- `_workspace.py` 只接受已完成的 plan，不能选择 arch、family、kid、redirect、dtype policy
  或 launcher ABI；
- a16w16 selector、planner、generated launcher 和 arch dispatch 是 family adapter；
- 只有未来出现 external two-stage workspace kernel 的其他 family 才新增 adapter；
- a8w8、a8w4 MoE 当前不需要 adapter，仓库中不存在 a4w4 实现。

### 用户指定机械检查的最终结果

原样执行：

```bash
rg -n "SplitkWsRegistry|opus_splitk_ws_|ws_handle" \
  csrc/opus_gemm aiter/ops/opus aiter/tuned_gemm.py

rg -n "hipMalloc|hipFree|hipHostMalloc" \
  csrc/opus_gemm/opus_gemm.cu \
  csrc/opus_gemm/codegen/gen_instances_gfx950.py \
  csrc/opus_gemm/codegen/gen_instances_gfx942.py \
  csrc/opus_gemm/codegen/gen_instances_gfx1250.py

rg -n "sys\\.path\\.(insert|append)" aiter/ops/opus csrc/opus_gemm
git diff --check
```

结果：前三条 `rg` 都是 exit 1、stdout 为空，即零命中；`git diff --check` 是 exit 0、
stdout 为空。额外检查 C++ allocator/registry 相关 include、stream capture/sync 变体亦为零
命中。唯一保留的 workspace 初始化入口是 Python deprecated no-op；C++ implementation 和
pybind 均不存在。legacy selector 只保留返回整数 kid 的 policy probe，不持有或 launch
workspace function pointer。

### canonical registry 全量覆盖

使用 `kernels_list -> get_kernel_instance(arch, "a16w16", kid)` 枚举全部 canonical a16w16
instance，并对每项核对 `kernel_needs_external_workspace()` 与
`plan_a16w16_workspace()` 的 plan/None 结果。对 workspace instance 使用其 `B_M/B_N/B_K`
和 `split_k=1` 实际构造 typed plan：

```text
gfx950  workspace=48,  non_workspace=92
gfx942  workspace=8,   non_workspace=14
gfx1250 workspace=496, non_workspace=0
errors=[]
```

这证明当前 registry 中三架构全部现有 two-stage split-K instance 都进入 actual-instance
workspace planner，而非只验证 heuristic 默认 kid。`_workspace.py` 单独扫描
`a16w16|gfx[0-9]|kid|split_k|launcher` 为零命中；selector 自身不分配 Tensor，唯一
`torch.empty` 集中在通用 allocator，并由 wrapper 在 resolved `LaunchConfig` 之后调用。

### 范围隔离与回归

从任务基线 `ca68b4f3` 到收尾前实现 HEAD，路径名属于 a8w8、a8w4、a4w4 或 MoE 的 tracked
diff 为零。Step 6 已记录 fresh generated a8w8 launcher/device/lookup 与基线逐字相同；
公共 API signature test 继续固定 a8w8 和 a8w4 stage1/stage2 参数，没有新增 workspace。

收尾后重跑：

```text
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py

159 passed, 14 skipped, 2 warnings in 4.46s
```

### gfx942 完成边界

现存 ISA artifact 重新核对：main `v_readfirstlane_b32` 为 `5 -> 5`，main VGPR/SGPR 为
`169/96 -> 169/96`，完整 reduce TU 的 readfirstlane 为 `276 -> 276`。旧间接 load 被
删除；124 个 reduce kernel 的 hardware VGPR block 不变、10 个改善、0 个回退。因此
wave-uniform address 语义和静态寄存器/occupancy 要求已闭环。

当前 8 张 GPU 均为 gfx950 MI355X，没有 gfx942。gfx942 split-K latency/throughput 的迁移
前后实测仍不能执行，也未写成通过。若“无性能回退”要求真实性能数据而不是 ISA/寄存器
证据，则它是任务一唯一剩余的外部硬件验收项；需要在 gfx942 机器运行基线与当前提交的同一
benchmark。

## 2026-08-11：按 actual kid 直接初始化 workspace 的结构精简（未提交）

- 日期：`2026-08-11`
- 分支：`splitk_to_torch_2`
- 修改基线：`2352c46c784d6ba3a0c71ff89b4bdb4c2fefa59f`
- 当前状态：tracked 修改尚未提交
- 合计：16 个 tracked 文件，新增 503 行，删除 713 行
- raw C++/pybind ABI：未修改，仍为 caller-owned optional Tensor + kid + split-K
- kernel ABI：未修改，仍为 direct workspace pointer
- 本节取代的仅是 Step 2/5/6 中独立 Python `WorkspacePlan`/planner/allocator 层；已提交
  selector、5/6 参数 generated dispatch、C++ validator 和 direct-pointer kernel 继续保留

### 用户确认的新设计意图

Step 1 至 Step 6 完成后，用户进一步明确：

1. workspace 不能根据 requested kid 或 architecture 粗粒度推导，必须根据 selector 最终
   得到的 `actual_kid` 生成；
2. heuristic 必须被完整考虑，但不应在 workspace 代码里重跑；
3. 不要增加专门的 workspace Python 文件，也不要保留偏重的通用 `WorkspacePlan`；
4. workspace dtype 应与 gfx942 一样由 exact kid 决定，并让 Python 与三架构 codegen 使用
   同一份 instance metadata；
5. generated C++ launcher 继续承担 caller-provided Tensor 的物理合同校验。

因此，本轮没有修改 `_selector_a16w16.py`：原 selector 已经正确执行
`explicit -> tuned CSV -> heuristic -> framework fallback`，并在返回 `LaunchConfig` 前完成
requested/actual kid、gfx942 redirect 和 split-K 解析。本轮修改的是 selector 之后的
workspace 初始化层和 dtype metadata 消费方式。

### BF16 能力复核与 gfx1250 two-stage 合同恢复

用户随后指出 gfx950/gfx1250 都有 bf16 能力。复核结果表明，旧总结把“bf16 输入/输出”与
“bf16 split-K partial workspace”混为一谈，同时遗漏了 gfx1250 PR #4246 feature 分支：

| 路径 | 当前/历史 kid | 物理 partial storage | 结论 |
|---|---|---|---|
| gfx950 当前 two-stage FlatMM | 200--223、1200--1223，共 48 个 | fp32 | A/B 为 bf16、Y 可 bf16/fp32，但没有现存 bf16-workspace kid |
| gfx1250 当前未提交修改 two-stage | 20000--20027、20100--20567，共 496 个 | bf16 | plain 28 + clusterlaunch 468，已恢复 #4246 的真实物理合同 |
| gfx1250 exact-kid two-stage code path | 当前 496 BF16；合成 FP32 路径也已验证 | bf16/fp32 | metadata 同步驱动 main、reduce、validator 和容量字节数 |
| gfx1250 PR #4246 fused | 21000--22377，共 1378 个 | bf16 780 / fp32 598 | 独立 family、per-kid dtype，且布局不同于当前 two-stage contract |

源码/历史依据：

- gfx950 原始提交 `29810587` 的 traits 明确要求 fp32 partial，reduce 参数为
  `const float* workspace`；当前 `splitk_reduce_gfx950.cuh` 仍固定 `D_WS=float`。因此不能因
  bf16 A/B/Y 能力就把现有 gfx950 workspace metadata 改成 bf16。
- gfx1250 feature 起点 `b32785d0` 在两个 two-stage main pipeline 中默认定义
  `OPUS_WS_BF16=1`，并让 reduce 模板携带匹配 `D_WS`；后续 Torch workspace 提交
  `dc2f4890`、shape/dtype 修复 `ea093a77` 和 PR head `25dd6281` 都保留了该物理路径。
- 该 feature 分支的旧 `splitk_workspace_dtype` 对 two-stage 仍默认 fp32，和实际 bf16
  writer/reader 不一致；fused family 则另用 `fuse_ws_dtype`。当前修改没有照抄旧 metadata，
  而是把现有 496 个 two-stage kid 显式标为 `bf16_t`，并同步 writer、reader、validator、
  tuner 和测试。
- fused 1378 个 kid 当时没有加入 `SPLITK_KIDS`，而且其 partial 布局不是当前 two-stage
  `[split_k, padded_M, padded_N]` 规则。若纳入本次迁移，必须建立显式 capability/registry
  和独立 shape 合同，不能继续用 kid 数值范围推断。

所以，gfx1250 不能再描述为 FP32-only。当前按 `actual_kid` 读取 metadata 的设计保持不变：
现有 496 个 two-stage kid 按 #4246 使用 BF16；底层 two-stage 模板仍能为未来 exact kid
生成 FP32。#4246 fused 的 1378 个 kid尚未实际合入；合并时必须把旧 `fuse_ws_dtype`
投影到统一的 `splitk_workspace_dtype`，加入显式 external-workspace registry，并实现其独立
tile-major shape/capacity 分支，不能只按 21000--22377 数值范围推断。

### 当前端到端流程

```text
gemm_a16w16_opus
  -> _validate_and_reshape
  -> select_launch_config
       explicit -> tuned CSV -> architecture heuristic -> framework fallback
       requested_kid -> redirect/legality/split resolver -> actual_kid
       allocation_split_k + launch_split_k
  -> fallback: _framework_a16w16
     or
     _launch_a16w16_with_torch_workspace(resolved config)
       -> _init_a16w16_workspace
            canonical instance = get_kernel_instance(..., config.actual_kid)
            capability = kernel_needs_external_workspace(..., actual_kid)
            tile/shape = instance.B_M/B_N/B_K + allocation_split_k
            dtype = instance.splitk_workspace_dtype
            caller Tensor passthrough or torch.empty
       -> _opus_gemm_a16w16_tune_raw(
            XQ, WQ, Y, bias, workspace,
            config.actual_kid, config.launch_split_k)
       -> generated non-workspace/workspace dispatch
       -> opus_validate_workspace physical-contract guard
       -> direct-pointer main/reduce kernels
```

heuristic 只负责产生 requested kid，随后与 explicit/CSV 请求一样进入同一个
`_build_launch_config()`。workspace init 不知道选择来源，也不需要知道 source 是
`explicit`、`tuned` 还是 `heuristic`；它只消费完全解析后的 `actual_kid`。

关键 redirect 例子保持为：

```text
gfx942, non-exact N
requested_kid=10210 (bf16 workspace metadata)
  -> actual_kid=10200 (fp32 workspace metadata)
  -> _init_a16w16_workspace reads kid 10200
  -> allocate fp32 workspace and launch kid 10200
```

### 本轮文件总表

| 状态 | 文件 | 行数变化 |
|---|---|---:|
| 删除 | `aiter/ops/opus/_workspace.py` | `+0/-186` |
| 删除 | `aiter/ops/opus/_workspace_a16w16.py` | `+0/-141` |
| 修改 | `aiter/ops/opus/gemm_op_a16w16.py` | `+66/-29` |
| 修改 | `csrc/opus_gemm/codegen/common.py` | `+19/-2` |
| 修改 | `csrc/opus_gemm/codegen/gen_instances_gfx1250.py` | `+65/-39` |
| 修改 | `csrc/opus_gemm/codegen/gen_instances_gfx942.py` | `+4/-15` |
| 修改 | `csrc/opus_gemm/codegen/gen_instances_gfx950.py` | `+12/-4` |
| 修改 | `csrc/opus_gemm/gen_instances.py` | `+20/-8` |
| 修改 | `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh` | `+7/-5` |
| 修改 | `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh` | `+7/-5` |
| 修改 | `csrc/opus_gemm/include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh` | `+17/-10` |
| 修改 | `csrc/opus_gemm/include/gfx1250/splitk_reduce_gfx1250.cuh` | `+37/-23` |
| 修改 | `csrc/opus_gemm/opus_gemm_common.py` | `+28/-10` |
| 修改 | `csrc/opus_gemm/opus_gemm_tune.py` | `+25/-9` |
| 修改 | `op_tests/test_opus_graph.py` | `+8/-14` |
| 修改 | `op_tests/test_opus_workspace.py` | `+188/-213` |

未跟踪的两份总结文档不计入以上 tracked numstat。

### 1. `aiter/ops/opus/_workspace.py`

状态：删除，`+0/-186`。

删除内容：

- `WorkspacePlan` dataclass；
- `checked_numel()`；
- Python `validate_workspace()`；
- `allocate_workspace()`；
- 该模块内的 dtype/shape/bytes/alignment 通用抽象。

删除原因不是取消安全检查，而是用户确认 a16w16 当前不需要独立通用层。分配前的结构检查
收进现有 a16w16 入口；caller Tensor 的 device/dtype/contiguous/alignment/capacity 最终
检查继续由 generated C++ launcher 完成。

### 2. `aiter/ops/opus/_workspace_a16w16.py`

状态：删除，`+0/-141`。

删除 `plan_a16w16_workspace()` 及其独立 family adapter。原有的 actual instance tile、
shape、dtype、gfx1250 batch 和 split-K K-tile 上限逻辑迁入
`gemm_op_a16w16.py::_init_a16w16_workspace()`，不再构造中间 plan 对象。

### 3. `aiter/ops/opus/gemm_op_a16w16.py`

状态：修改，`+66/-29`。

#### import 与 dtype 映射

删除：

```python
WorkspacePlan
allocate_workspace
validate_workspace
plan_a16w16_workspace
```

新增私有 token 映射：

```python
_WORKSPACE_DTYPES = {
    "bf16_t": torch.bfloat16,
    "fp32_t": torch.float32,
}
```

该映射只把 exact-kid metadata token 转为 Torch dtype，不包含 architecture 或 kid policy。

#### `_prepare_a16w16_workspace()` 替换为 `_init_a16w16_workspace()`

新 helper 返回 `Tensor | None`，不再返回 `(WorkspacePlan, Tensor)`。行为为：

- framework fallback 或缺少 actual kid 时拒绝；
- 用 `(config.arch, config.family, config.actual_kid)` 取得 canonical instance；
- 按 actual kid 查询 external-workspace capability；
- non-workspace actual kid 要求 caller `workspace is None` 并返回 `None`；
- 从 XQ/Y 读取 batch/M/N/K，从 actual instance 读取 `B_M/B_N/B_K`；
- 要求 `allocation_split_k > 0`，并在分配前拒绝超过
  `ceil(K / actual_instance.B_K)` 的值；
- 计算 padded M/N；
- gfx942/gfx950 使用四维
  `[split_k, batch, padded_M, padded_N]`；
- gfx1250 使用三维 `[split_k, padded_M, padded_N]`，继续拒绝 batch != 1；
- 从 `actual_instance.splitk_workspace_dtype` 读取 bf16/fp32 token；
- 对 shape 元素数和 dtype 字节数做逐 extent overflow 检查；
- caller 提供 Tensor 时原样返回，由 raw/generated C++ 做最终物理合同验证；
- caller 未提供时直接执行
  `torch.empty(shape, dtype=dtype, device=XQ.device)`。

`_launch_a16w16_with_torch_workspace()` 改为调用该 init，再把 Tensor、actual kid 和
launch split-K 交给 raw binding。高层 `gemm_a16w16_opus()` 仍只执行一次 selector，
explicit/CSV/heuristic 三种 OPUS source 继续汇合到相同 resolved-config 路径。

### 4. `csrc/opus_gemm/opus_gemm_common.py`

状态：修改，`+19/-2`。

`OpusGemmInstance.splitk_workspace_dtype` 从：

```python
splitk_workspace_dtype: str = "fp32_t"
```

改为：

```python
splitk_workspace_dtype: str | None = None
```

语义变为：non-workspace kid 可以不声明；external-workspace kid 必须显式声明物理 storage
dtype。为此前依赖默认 fp32 的 gfx950 和 gfx942 fp32 split-K factory 补充
`splitk_workspace_dtype="fp32_t"`；既有 gfx942 bf16 变体继续显式声明 `"bf16_t"`。gfx1250
factory 则显式声明 `"bf16_t"`，使当前 28 个 plain + 468 个 clusterlaunch two-stage kid与
#4246 的 writer/reader 合同一致。

在 `SPLITK_KIDS` 构造后新增全量 invariant：每个 workspace kid 的 dtype 必须属于
`{"bf16_t", "fp32_t"}`，缺失或未知 token 在 import/codegen 前立即失败。这避免新增 kid
静默继承架构级默认值。

### 5. `csrc/opus_gemm/codegen/common.py`

状态：修改，`+17/-0`。

新增 `_SPLITK_WORKSPACE_TYPES` 和 `splitk_workspace_type(k)`，统一把 instance metadata
映射为：

| metadata | C++ storage token | pointer type | Aiter dtype token |
|---|---|---|---|
| `bf16_t` | `bf16_t` | `__bf16` | `AITER_DTYPE_bf16` |
| `fp32_t` | `fp32_t` | `float` | `AITER_DTYPE_fp32` |

缺失/未知 token 抛 `ValueError`，不提供 fp32 fallback。该 helper 是三架构 generator 的
共享入口。

### 6. `csrc/opus_gemm/codegen/gen_instances_gfx942.py`

状态：修改，`+4/-15`。

删除本文件私有 `_splitk_workspace_types()` 和 `_uses_bf16_workspace()`；它们以前对缺失
metadata 默认 fp32。改为调用 `codegen.common.splitk_workspace_type(k)`，并由返回的
storage token 推导既有 `bf16ws` 分支。traits、reduce pointer type 和
`opus_validate_workspace()` expected dtype 现在来自同一个 exact-kid metadata。

### 7. `csrc/opus_gemm/codegen/gen_instances_gfx950.py`

状态：修改，`+12/-4`。

`gen_flatmm_splitk_instance()` 现在首先读取 `splitk_workspace_type(k)`：

- traits 中的 workspace storage type 使用 metadata token；
- `opus_validate_workspace()` expected dtype 使用 metadata 对应的 Aiter token；
- 当前 gfx950 main/reduce 物理实现只支持 fp32，因此对非 fp32 声明显式抛
  `ValueError`。

此 guard 表示“kid metadata 是权威，但不能声明底层尚未实现的能力”，不再由 arch 在
Python planner 中硬编码 dtype。

### 8. `csrc/opus_gemm/codegen/gen_instances_gfx1250.py`

状态：修改，`+65/-39`。

cluster/TDM split-K generator 读取 shared exact-kid metadata，并把同一结果同时用于：

- traits 的 `D_WS`；
- `opus_validate_workspace()` 的 expected Aiter dtype；
- reduce kernel 的 `D_WS_` 模板参数；
- reduce geometry：BF16 使用 `VEC=8/BLOCK=128`，FP32 使用 `VEC=16/BLOCK=64`。

删除了 gfx1250 FP32-only guard。BF16/FP32 Y 和 bias 分支都显式携带 workspace pointer
type。dedicated reduce TU 现在包含：

- FP32 workspace 的 baseline matrix；
- BF16 workspace 的 BF16/FP32 Y、with/without bias matrix；
- `BF16 Y + FP32 bias` 的 BF16 与 FP32 workspace 两种缺省矩阵外组合。

这样当前 496 个 BF16 two-stage kid和可能的 FP32 two-stage exact kid可共享同一生成器，
不再由 architecture 决定 storage。#4246 的 FP32 fused kid仍使用独立 fused generator，
合并边界见本节前述说明。

### 9. `csrc/opus_gemm/gen_instances.py`

状态：修改，`+20/-8`。

- gfx1250 baseline reduce 显式实例现在明确写出最后一个 `D_WS_=float`，不再只依靠模板
  默认参数表达 FP32 storage；
- per-arch host TU 的 reducer forward declaration增加 gfx1250 `typename D_WS_`，与 generated
  launcher 和 reducer 定义的七个模板参数一致；
- gfx942/gfx950 仍保持原有六参数 reducer 声明，未被 gfx1250 ABI 扩展污染。

### 10. gfx1250 traits、main writer 与 reducer

涉及四个文件：

- `opus_gemm_traits_a16w16_gfx1250.cuh`；
- `opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh`；
- `opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh`；
- `splitk_reduce_gfx1250.cuh`。

物理合同修改为：

- traits 允许 `D_WS` 为 BF16 input storage 或 FP32 accumulator storage；
- plain/clusterlaunch main 都先执行 `opus::cast<DataWS>(reg_c)`，再把四个 exact-type 元素
  写入 `[split, padded_M, padded_N]`；
- reducer 新增 `D_WS_` 模板参数，从 matching BF16/FP32 workspace 读取后统一转成 FP32
  累加；
- VEC=8 与 VEC=16 都有完整 OOB tail store 分解；
- writer、reader 和 buffer descriptor 的字节数都使用 `sizeof(DataWS/D_WS)`。

### 11. `csrc/opus_gemm/opus_gemm_tune.py`

状态：修改，`+25/-9`。

`_kid_uses_bf16_workspace()` 删除
`getattr(..., "splitk_workspace_dtype", "fp32_t")` 的隐式默认，直接读取
`k_inst.splitk_workspace_dtype`。另外：

- gfx942 的 BF16 exact-N/Y 限制只作用于 `arch_prefix == "gfx942"`，不会误伤 gfx1250；
- gfx1250 BF16 workspace 允许 BF16 或 FP32 Y；
- gfx1250 plain/clusterlaunch per-slice buffer-resource 字节数按 exact dtype 使用 2/4 字节；
- `candidate_splitK()` 的总 workspace 4 GiB 上限也按 exact dtype 使用 2/4 字节，不再固定
  乘 4。

### 12. `op_tests/test_opus_workspace.py`

状态：修改，`+188/-213`。

测试从“独立 WorkspacePlan/allocator/validator”改为“resolved actual-kid init + C++ raw
防线”：

- 删除对两个已删除模块和 `WorkspacePlan` API 的 import/测试；
- 新增所有 `SPLITK_KIDS` 显式声明 bf16/fp32 storage dtype 的全量断言；
- 直接调用 `_init_a16w16_workspace()` 验证 gfx950、gfx942 fp32/bf16 和当前 checkout
  gfx1250 BF16 kids 的 actual-kid tile、shape、dtype；
- 精确断言 gfx1250 two-stage registry 为 plain 28 + clusterlaunch 468，且 496 个全部 BF16；
- 用合成 gfx1250 FP32 exact kid验证同一 codegen 会切换 traits、validator、reduce
  geometry 和 `D_WS`；
- 生成完整单-kid host/main/reduce TU，断言 host forward declaration含 `D_WS_`，并断言
  BF16/FP32 mixed-bias reducer specialization 都存在；
- 显式验证 gfx1250 BF16 workspace kid仍接受 FP32 Y；
- 验证 gfx942 `10210 -> 10200` 后使用 10200 的 fp32 dtype/shape；
- 保留 non-workspace、gfx1250 batch、split-K K-tile limit、显式 Tensor 复用和每次调用
  独立 Tensor 覆盖；
- production fake-raw 继续断言自动 workspace、actual kid 与 launch split-K；
- GPU raw fixture 不再先构造 plan，而是复用真实 init 分配的 Tensor；
- missing/dtype/device/contiguous/alignment/capacity 负例仍以 generated C++ validator 为
  验收对象。

### 13. `op_tests/test_opus_graph.py`

状态：修改，`+8/-14`。

- graph capture 测试不再 monkeypatch 已删除的 `allocate_workspace()`，改为在 raw boundary
  记录实际传入 workspace 指针；
- replay 仍断言 Python/raw 入口只在 capture 时进入一次；
- 无全局 Tensor cache 扫描只检查现存 `gemm_op_a16w16` 模块，不再 import 已删除模块；
- graph、双 stream、weakref 生命周期和 deprecated init 的原语义保持不变。

### 验证状态

已完成且通过：

1. 相关 Python runtime、metadata、codegen 和测试文件 `py_compile`；
2. `git diff --check`；
3. 全仓 Python 源码扫描确认没有残留 `_workspace_a16w16`、`WorkspacePlan`、
   `plan_a16w16_workspace`、`allocate_workspace` 或旧 prepare helper 的 import/调用；
4. 仅选择不执行 GPU kernel 的 dispatch/workspace/graph 用例：

```text
141 passed, 18 deselected, 2 warnings in 3.68s
```

18 项 deselect 是明确排除的 raw GPU、graph replay 和双-stream 用例。该 141-case 结果覆盖：

- explicit/CSV/heuristic/fallback 选择与优先级；
- requested/actual kid 和 gfx942 redirect；
- split-K resolver；
- exact-kid tile、shape 和 dtype；
- 三架构 workspace/non-workspace capability；
- 所有 workspace kid 显式 dtype invariant；
- gfx1250 28 plain + 468 clusterlaunch 的 BF16 metadata 全量断言；
- gfx1250 BF16 workspace + FP32 Y selector 合法性；
- 合成 FP32 exact kid 的 generator/host/reduce 矩阵；
- call-scoped 独立 Tensor、无 Python Tensor cache；
- fake raw boundary 的 workspace、actual kid 和 launch split-K。

5. fresh gfx1250 codegen 与 HIP syntax：

- `/tmp/opus-gfx1250-codegen.9Jtck6`：当前 BF16 registry 的 6-kid默认 subset；generated
  `all_instances_host_gfx1250.cu`、6 个 plain main device TU 和 typed reduce TU 均通过
  `hipcc -std=c++20 -O3 --offload-arch=gfx1250 -fsyntax-only`；
- `/tmp/opus-gfx1250-cluster.sk2CCK`：代表 clusterlaunch BF16 main device TU 通过；
- `/tmp/opus-gfx1250-fp32.TsbHXs`：合成 FP32 exact kid生成
  `AITER_DTYPE_fp32` validator、FP32 traits、VEC=16 reduce launch；main 和 reduce device TU
  均通过相同 target 的 HIP syntax；
- reduce TU 机械检查确认同时存在显式 `D_WS=float` 与 `D_WS=__bf16` 矩阵；host TU 确认
  forward declaration含 `typename D_WS_`；
- 只有仓库既有的 `opus.hpp` deduction-guide attribute、`aiter_tensor.h` ignored-result 和
  `--hip-link` unused warning，没有本轮 compile error。

未计为验证结果的中断：`2026-08-11 02:27:10 UTC` 发起

```text
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py
```

约 4 秒后系统记录 AMD GPU queue eviction 和 Docker veth 退出，旧容器/会话随即终止；
rollout 只有 tool call，没有对应 tool output 或 final answer。未发现 OOM、segfault 或 core
dump，因此只能确认运行环境被终止，不能证明 pytest pass/fail，也不能把 pytest 断言为
容器退出的根因。

重建环境后，`rocm-smi` 快照显示 8 张 GPU 均为 100% busy、每卡约 92% VRAM；KFD 列出
8 个当前容器不可见的 UNKNOWN 进程，当前容器内没有残留 pytest。为避免干扰归属不明的
宿主机/其他容器任务，本轮没有继续运行 GPU 用例，也没有杀进程。完成 gfx1250
codegen/HIP syntax 后再次复查，8 卡瞬时 GPU use 为 0%，但 VRAM 仍为 94%--95%，同一组
8 个 `UNKNOWN` KFD 进程仍不在当前容器 PID namespace；因此仍只做离线交叉编译，不启动
GPU 数值、graph 或性能测试。

### 提交前仍需完成

1. 当前 496 个 two-stage BF16 kid已恢复；#4246 的 1378 个 fused kid仍需单独合入。合入时
   必须登记 780 个 BF16 / 598 个 FP32 exact metadata、external-workspace capability，并实现
   `(tile, split_k-1, B_M, B_N)` 物理次序对应的独立 shape/capacity 分支；
2. GPU 可用后重跑完整 focused suite：

```text
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py
```

3. gfx1250 本轮路径已完成 fresh syntax；提交前仍应 fresh 生成 gfx942/gfx950/gfx1250
   组合默认 subset，并对全部 generated host/device/reduce TU执行目标架构 HIP syntax，
   确认 per-arch forward declaration在组合构建中也无回归；
4. 在 gfx950 复验数值、raw exact/short-one 和全部物理合同负例、graph replay、双 stream、
   生命周期与跨 device；
5. 更新 `aiter/ops/opus/README.md` 与 `csrc/opus_gemm/README.md`，删除已经失效的
   `WorkspacePlan`/独立 planner 描述，改成单一 init + exact-kid metadata；
6. 重跑 legacy-symbol/allocator/`sys.path`/scope-isolation 机械检查；
7. 审阅并提交当前 16 个 tracked 文件；不要 amend Step 1 至 Step 6 的历史提交；
8. gfx942/gfx1250 实机和 gfx942 真实性能仍是外部硬件 follow-up，边界与 Step 6 相同。

## 2026-08-11：完成 gfx1250 PR #4246 fused family 迁入（未提交）

- 日期：`2026-08-11`
- 分支：`splitk_to_torch_2`
- 基线 HEAD：`2352c46c784d6ba3a0c71ff89b4bdb4c2fefa59f`
- 当前状态：代码与文档修改尚未提交
- raw C++/pybind ABI：未修改，仍为 optional caller-owned workspace + actual kid + split-K
- 本节完成上一节“提交前仍需完成”的第 1 项；“1378 个 fused kid尚未合入”已经是历史状态
- 最终 tracked diff：17 个文件，`+1366/-731`；另新增 1 个未跟踪 fused pipeline 文件

### 最新直接框架

```text
public a16w16 API
  -> explicit / tuned CSV / heuristic / framework fallback
  -> resolved actual_kid + resolved split-K
  -> exact-kid registry
       non-workspace -> direct launch
       two-stage     -> split-major Torch workspace -> main -> standalone reduce
       fused         -> tile-major Torch workspace  -> clustered main + in-kernel reduce
  -> generated physical-contract validator
  -> launch
```

`actual_kid` 同时决定 workspace capability、`splitk_workspace_dtype`、tile、shape family和 launcher。
Python 只做调用级 Tensor 分配与安全的结构计算；generated C++ 对 caller Tensor 做最终物理合同
校验。当前实现没有全局 workspace cache，也没有恢复 `WorkspacePlan` 或旧 `fuse_ws_dtype`。

### 最终架构口径

#### gfx950

gfx950 当前只有 48 个 FlatMM two-stage kid 使用 external Torch workspace：

```text
200--223, 1200--1223
workspace dtype = FP32
```

其他 split-barrier、persistent、wave/cooperative 和 atomic-accumulate 路径不发布 external
partial workspace。它们可能在算法意义上使用 split-K 或跨 wave/WG 累加，但没有 caller-owned
Torch workspace，因此没有“普通 split-K workspace 应保持 BF16 还是 FP32”的 dtype 选择。

canonical 汇总：

```text
gfx950 external-workspace a16w16 = 48  (FP32 48)
gfx950 non-workspace a16w16      = 92
```

#### gfx1250

gfx1250 当前 external-workspace registry 已完整包含：

| family | kid | 数量 | BF16 workspace | FP32 workspace | reduce 方式 |
|---|---:|---:|---:|---:|---|
| plain two-stage | 20000--20027 | 28 | 28 | 0 | 独立 reduce kernel |
| clusterlaunch two-stage | 20100--20567 | 468 | 468 | 0 | 独立 reduce kernel |
| fused | 21000--22377 | 1378 | 780 | 598 | 同一 clustered kernel 内 reduce |
| 合计 |  | 1874 | 1276 | 598 |  |

这表示 gfx1250 现在确实同时申请 BF16 和 FP32 workspace，但不是每个 kid同时支持两种类型；
每个 exact kid只声明一种物理 storage dtype，Python 分配和 C++ validator都按该 kid 的
`splitk_workspace_dtype` 执行。

### fused 的物理合同

two-stage 继续使用：

```text
[runtime_split_k, padded_M, padded_N]
```

fused 使用独立 tile-major 合同：

```text
[num_tiles_m, num_tiles_n, fuse_split_k - 1, B_M, B_N]
```

其执行过程为：

1. cluster.x 的 `SplitK` 个 workgroup分别计算 K slice；
2. 前 `SplitK-1` 个 WG把 exact `D_WS` partial 发布到 caller-owned workspace；
3. 最后一个 WG经 cluster barrier读取这些 partial；
4. partial 统一转为 FP32 累加，最后一次 cast 后直接写 BF16/FP32 Y；
5. 不再启动第二个 reduce kernel。

`SplitK` 与 N-direction cluster peer count都是 exact-kid compile-time 属性。runtime 或 tuned CSV
中的 `splitK` 不得改变 fused 执行和容量；selector 将 `allocation_split_k`、
`launch_split_k` 和 `effective_split_k` 都解析为该 kid 的 `fuse_split_k`。历史字段
`fuse_m_cluster` 为兼容 #4246 名字而保留，当前物理含义是 N-tile peer count。

### 1. `csrc/opus_gemm/opus_gemm_common.py`

状态：修改。

新增/修改内容：

- `OpusGemmInstance` 新增 `fuse_split_k` 与兼容字段 `fuse_m_cluster`；
- instance name 增加 `skfuse`、N peer、SplitK、workspace dtype 和 prefetch/WG 信息，确保每个
  compile-time 变体的 symbol 唯一；
- 新增 `_a16w16_splitk_fuse_gfx1250()` factory；
- 新增 `gfx1250_splitk_fuse_kernels_list`、`GFX1250_SPLITK_FUSE_KIDS` 和
  `GFX1250_SPLITK_FUSE_KID_OF`；
- 确定性生成 kid `21000..22377`，并在 import 时断言总数 1378、BF16 780、FP32 598；
- fused dtype 直接写入共享 `splitk_workspace_dtype`，没有恢复 PR 旧 `fuse_ws_dtype`；
- 1378 个 kid加入 `kernels_list` 与 `SPLITK_KIDS`，因此 external-workspace capability来自
  显式 registry membership，而非 `21000..22377` 数值范围；
- BF16 family覆盖 SplitK 2..15，FP32 family覆盖 2..8；同时受 16-WG cluster budget、
  N-peer 1..5 和 reduce-ring LDS容量约束；
- 全量 dtype invariant继续要求每个 external-workspace kid显式声明 `bf16_t` 或 `fp32_t`。

### 2. 新增 fused pipeline

新增文件：

```text
csrc/opus_gemm/include/gfx1250/
  opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_fuse_gfx1250.cuh
```

实现内容：

- `__cluster_dims__(SplitK, NClusterWg, 1)` 的 single-kernel split-K；
- A 在 N peers 之间通过 cluster TDM multicast共享；
- non-last WG按 exact `DataWs` 执行 lane-contiguous partial store；
- last WG通过 bounded LDS ring分批 TDM stage `SplitK-1` 个 partial，始终在 FP32 累加；
- 支持 BF16/FP32 workspace 和 BF16/FP32 Y；
- ragged M使用 bounded C descriptor，N 必须完整填充 `B_N` tile和 N cluster；
- `SplitK <= ceil(K/B_K)`，避免空 split WG进入 cluster barrier；
- physical offset 与 Python/C++ capacity一致：

```text
((tile_m * num_tiles_n + tile_n) * (SplitK - 1) + partial)
    * (B_M * B_N)
```

- PR #4246 原实现依赖旧 `opus::tdm/make_tdm` API；迁入时改为当前树的
  `opus::tdm_window` API，没有恢复已经删除的旧接口。

### 3. gfx1250 traits 与 kargs

文件：

```text
csrc/opus_gemm/include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh
```

新增 `opus_gemm_splitk_fuse_kargs_gfx1250`，包含 direct workspace pointer、A/B/Y/bias、
实际 M/N/K、stride 和 tile counts。traits 中 `D_WS` 继续是唯一 physical workspace type，
同时供 two-stage 与 fused 使用。

fused 当前保留 #4246 round-1 bias kernel合同：contiguous BF16 `[N]`。这比公共 API允许的
BF16/FP32、`[N]`/`[batch,N]` 更窄，安全选择策略见后文。

### 4. `csrc/opus_gemm/codegen/gen_instances_gfx1250.py`

状态：修改。

新增 fused tag 的 pipeline/traits/kernel/kargs maps 和 `gen_splitk_fuse_instance()` emitter：

- host workspace-dispatch specialization继续使用 `<fp32_t>` token；
- traits 中的真实 `D_WS` 来自 exact-kid `splitk_workspace_dtype`；
- launcher验证 batch=1、M/N/K、K 偶数、N tile、N cluster fill和 compile-time SplitK K-tile
  上限；
- `opus_validate_workspace()` 检查 device、exact dtype、contiguous、16-byte alignment 和
  tile-major exact capacity；
- launcher忽略 runtime `splitK` 对容量的影响，只 launch exact-kid compile-time geometry；
- Y dtype在 host 侧选择 BF16/FP32 device specialization；
- 每个 fused kid生成一个 host impl header和两个 device TU；
- fused emitter不登记 standalone reduce specialization。

### 5. shared codegen 与总生成器

修改：

```text
csrc/opus_gemm/codegen/common.py
csrc/opus_gemm/gen_instances.py
```

- fused tag加入 a16w16 emit registry；
- fused 加入 `SPLITK_TAGS`，因此进入现有六参数 external-workspace manifest/lookup；
- manifest仍按 generated table membership区分五参数 non-workspace 与六参数 workspace ABI；
- host/device TU 生成支持 fused 的 host launch stub和 per-output device specialization；
- fused-only subset不会生成 `splitk_reduce_gfx1250.device.cu`；
- CLI `--kernel_tag a16w16_clusterlaunch_tdm_splitk_fuse` 可生成该 family；
- lookup 中 1378 个 fused kid全部进入 gfx1250 workspace table，没有数值范围分支。

### 6. Python selector 与 workspace 分配

修改：

```text
aiter/ops/opus/_selector_a16w16.py
aiter/ops/opus/gemm_op_a16w16.py
```

selector 对 fused 增加：

- batch=1；
- K 偶数；
- `N % B_N == 0`；
- `num_tiles_n % n_cluster == 0`；
- `fuse_split_k * n_cluster <= 16`；
- `fuse_split_k <= ceil(K/B_K)`；
- runtime/CSV splitK重写为 exact-kid compile-time value。

`_init_a16w16_workspace()` 在取得 resolved `actual_kid` 后按 tag分两条 shape 公式：

- two-stage：`[allocation_split_k, padded_M, padded_N]`；
- fused：`[num_tiles_m, num_tiles_n, fuse_split_k-1, B_M, B_N]`。

两条路径都从同一个 `splitk_workspace_dtype` 映射 Torch BF16/FP32，caller Tensor继续原样传入
generated C++ final validator；自动路径使用 `torch.empty` 分配每调用独立 Tensor。

### 7. tuner 与 subset candidate

文件：`csrc/opus_gemm/opus_gemm_tune.py`。

新增内容：

- fused exact-kid candidate不扫描 runtime SplitK；
- 先按 tile occupancy选 bounded tile set，再取 baseline/max feasible N cluster；
- 对每个 `N-cluster x workspace dtype` 从实际 exact-kid registry 独立按 occupancy选最多 3 个
  compile-time SplitK。BF16 当前覆盖 2--15、FP32 覆盖 2--8，独立选择避免共用 top-N 在
  大 K、小 grid时只命中高 SplitK BF16 而错误清空合法 FP32 候选；
- 候选扫描上限由 `GFX1250_SPLITK_FUSE_KID_OF` 自动推导为
  `GFX1250_FUSE_MAX_SPLITK`（当前为 15），避免极大 K 触发与 registry 无关的超长 range；
- `candidate_splitK()` 对 fused只返回 baked SplitK，使 CSV自描述；
- `kid_rejects_shape()` 镜像 N fill、cluster budget、K tile和 tile-major容量约束；
- fused kids加入 tuner 的 a16w16 kernel map和 subset codegen sidecar路径；
- `OPUS_TUNE_NO_FUSE=1` 可在隔离调试时关闭 fused candidates。

#### bias 安全门

tuned CSV key只记录 `bias=true/false`，不能记录 bias dtype/shape。若允许 fused bias，使用 BF16
`[N]` 调出的 row可能被 FP32 或 `[batch,N]` bias重放并在 launcher处失败。因此当前策略为：

- `bias=False`：fused 正常进入 candidate/selector；
- `bias=True`：tuner不产生 fused candidate，selector也拒绝 explicit/tuned fused kid；
- gfx1250 two-stage继续承担公共 BF16/FP32、`[N]`/`[batch,N]` bias 合同。

这不是 workspace dtype 限制，而是 tuned schema无法表达 fused round-1 bias 物理合同的安全
门。未来若 CSV/selector携带 exact bias dtype/shape，可单独放开 compatible fused bias。

### 8. 测试

`op_tests/test_opus_workspace.py` 新增/更新：

- 1378 / 780 / 598 registry exact count与 kid range；
- fused 不存在 `fuse_ws_dtype` 第二字段；
- BF16/FP32 fused codegen、validator token、无 standalone reduce TU；
- pipeline使用 `tdm_window` 且不使用 `make_tdm`；
- tile-major Torch shape和 compile-time SplitK不受 runtime值影响；
- selector使用 baked SplitK；
- public boolean-bias selector拒绝 fused；
- tuner bias candidate集合不含 fused；
- tuner 按 workspace dtype独立选择 compile-time SplitK；回归形状
  `M=16,N=32,K=4096,CU=256,tile=16x32x128` 同时保留 BF16 SplitK 13--15 与 FP32
  SplitK 6--8，并满足 16-WG cluster budget；
- gfx1250 BF16 workspace仍允许 FP32 Y；
- 既有 two-stage、gfx942 redirect、gfx950 FP32和 call-scoped Tensor覆盖保持。

### 最终验证

#### Python 与 CPU

```text
python -m py_compile ...
git diff --check

pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  -k 'not raw_cpp and not graph_capture and not two_streams'

149 passed, 18 deselected, 2 warnings in 3.69s
```

18 项 deselect 是明确排除、没有执行的真实 GPU raw、graph replay和双-stream用例，不是失败；
本次实际执行的选择集为 149 passed、0 failed。

#### canonical registry

```text
gfx950  workspace=48,   dtype={fp32: 48},              non_workspace=92
gfx942  workspace=8,    dtype={bf16: 3, fp32: 5},      non_workspace=14
gfx1250 workspace=1874, dtype={bf16: 1276, fp32: 598}, non_workspace=0
```

#### 全量 fused codegen

目录：`/tmp/opus-all-fused-codegen.4G1uK3`。

```text
1378 impl headers
2756 device TUs
1378 gfx1250 workspace lookup rows
0 standalone reduce TUs
```

该检查证明全 registry可生成，但没有把全部 2756 个 device TU逐一交叉编译。

#### fresh 组合 HIP syntax

目录：`/tmp/opus-fused-combined.yOZtal`。subset包含：

- 6 个 gfx1250 heuristic two-stage plain kid；
- 1 个 two-stage clusterlaunch kid；
- BF16 workspace fused + BF16/FP32 Y；
- FP32 workspace fused + BF16/FP32 Y；
- 128x128 大 tile、FP32 workspace、`SplitK=8`、`n_cluster=2`，即 16-WG cluster边界。

通过 gfx1250 `hipcc -fsyntax-only` 的文件：

```text
all_instances_host_gfx1250.cu
opus_gemm.cu
opus_gemm_pybind.cu
14 generated main/fused/reduce device TUs
```

只有仓库既有的 `opus.hpp` deprecated attribute、`aiter_tensor.h` ignored-result和
`--hip-link` unused warning，没有本轮 compile error。

#### gfx950 实机 focused suite

在两张明确空闲的 MI355X/gfx950上执行；物理 GPU 4、5通过 `HIP_VISIBLE_DEVICES=4,5`映射为
测试进程内 device 0、1。GPU 4承担主要 kernel、graph和双stream测试，GPU 5用于跨device
workspace拒绝测试：

```text
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py

162 passed, 14 skipped, 2 warnings in 4.39s
```

结果为 `0 failed`。14项均因当前硬件不是 gfx942/gfx1250而跳过。gfx950实际执行并通过：

- split-K kid 200的 BF16/FP32 Y数值对 FP32 golden；
- gfx950 bias数值与 dtype拒绝规则；
- exact typed workspace、少1 element、missing/wrong dtype/noncontiguous/misaligned；
- workspace跨device拒绝与 non-workspace kid必须传 `None`；
- 不调用旧 prewarm的 graph capture和三组输入 replay；
- 两个stream持有不同调用级 workspace并得到正确结果。

测试前后 `rocm-smi`确认物理 GPU 4、5从0%利用率/0%显存开始，并在进程退出后恢复0%/0%；
没有测试残留 KFD进程，也没有访问或终止 GPU 0--3上的既有进程。

### 未验证边界

- 没有运行 gfx1250 fused GPU 数值；
- gfx950 graph capture/replay和双stream已通过；gfx942/gfx1250对应路径仍未实机运行；
- 没有运行 gfx1250 fused graph capture/replay或双stream；
- 没有运行 fused 性能调优/benchmark；
- 没有逐一编译全量 2756 个 fused device TU；
- gfx942/gfx1250 实机和 gfx942 真实性能仍需对应硬件验收。

未知 GPU/KFD 进程仍占用物理 GPU 0--3，本轮没有终止或干扰这些进程。当前只把 gfx950
focused GPU suite写为实机通过；gfx942/gfx1250仍只具有 CPU、codegen和交叉语法证据，不能
表述为对应 GPU数值、graph、并发或性能通过。

### 当前后续

1. GPU 资源归属明确且可用时，在 gfx1250 运行 BF16/FP32 workspace × BF16/FP32 Y 的 fused
   数值矩阵，并覆盖 graph、双 stream和 bias=False tuner winner；
2. 若要求放开 fused bias，先扩展 tuned key/selector以携带 exact bias dtype/shape，再允许
   contiguous BF16 `[N]`，不能只删除当前安全门；
3. 审阅并提交当前工作树；不要 amend Step 1 至 Step 6 的历史提交。

## 2026-08-11：切换任务二前冻结任务一（gfx950 140-kid 全量记录）

用户决定先冻结并保存任务一当前进度，再切换任务二。本节追加事实记录，不覆盖前面的历史；
“冻结”不等于任务一剩余硬件、mono FP32归因和性能验收已经通过。

### 新增全量测试入口

新增未跟踪测试文件：

```text
op_tests/test_opus_gfx950_exhaustive.py
```

文件共305行，SHA-256：

```text
fc9e792ff1dec9c9964523fda477eba2287ccfcfea5b6db52762d00a75dc30fc
```

测试是 opt-in、可分片的 gfx950 release/acceptance sweep：

- 从最终 canonical `kernels_list`和 `get_kernel_instance("gfx950", "a16w16", kid)`枚举，
  不手写140个 kid；
- collection invariant固定为140 total、48 workspace、92 direct；
- workspace集合固定为 `200--223 | 1200--1223`，并断言当前 physical dtype全为 FP32；
- `OPUS_GFX950_SHARD_INDEX/COUNT`按 canonical ordinal稳定分片；
- 每个 workspace kid使用合法 exact tile和 `K=32*B_K`、`splitK=2`，运行 caller复用的
  BF16 Y、FP32 Y和自动分配 BF16 Y；workspace预填 NaN，验证 main完全覆写实际 partial，
  同时检查shape、dtype、device、contiguous、16-byte alignment、指针复用和 weakref释放；
- 每个 direct kid使用 `K=2*B_K`，运行 BF16/FP32 Y，并临时拦截 `gemm.torch.empty`，证明
  non-workspace路径不会分配 external workspace；
- 所有结果对 FP32 `torch.bmm` golden；BF16容差 `rtol=0.03, atol=0.5`，FP32容差
  `rtol=1e-3, atol=0.05`。

### 隔离全 kid构建

使用独立目录：

```text
/tmp/aiter-gfx950-current.NtJydE
```

sidecar先写入全部140个 canonical gfx950 a16w16 kid，再由生成器自动保留2个必需 a8w8项。
最终 sidecar共142项，SHA-256：

```text
b43395710e4d99e2e4ed5807dc495a6312e435b056d5f475d088496ff830bdf7
```

单进程构建成功并通过 kid 200烟测。生成/二进制核对结果：

```text
BF16 direct dispatch: 92
FP32 direct dispatch: 92
workspace dispatch:   48
offload archs:         gfx950 only
```

构建完成后才启动并行 shard，没有四个进程竞争同一个 JIT build。

### 四卡全量执行与结果

physical GPU 4--7各运行一个35-kid shard，核心环境为：

```text
HIP_VISIBLE_DEVICES=<4..7>
GPU_ARCHS=gfx950
AITER_JIT_DIR=/tmp/aiter-gfx950-current.NtJydE
OPUS_GFX950_EXHAUSTIVE=1
OPUS_GFX950_SHARD_COUNT=4
OPUS_GFX950_SHARD_INDEX=<0..3>
```

结果：

| shard | pytest摘要 | 失败 kid | log SHA-256 |
|---|---:|---|---|
| 0 | `32 passed, 3 failed` | 1400、1404、6401 | `4699111947edbc08109868464fc769d3784077f3da172de4b2b6954a4a15a2ae` |
| 1 | `33 passed, 2 failed` | 1401、6402 | `eeb19347d73a60950f1c49e6074605d9fdf700d9e914269fd42c8e631942513d` |
| 2 | `33 passed, 2 failed` | 1402、6403 | `30d1135a27e9e8a53d194c9eac0f9476ed363e48acc9823f08c4c3c3a236c787` |
| 3 | `32 passed, 3 failed` | 1403、6400、6404 | `8570df346ae8ebd84007ad456a6a10acdd8c77ddd0c624f9d665c03b25b56705` |
| 合计 | `130 passed, 10 failed` | 1400--1404、6400--6404 | — |

原始日志和 JUnit XML位于：

```text
/tmp/aiter-gfx950-current.NtJydE/results/shard{0,1,2,3}.{log,xml}
```

这些路径属于临时存储；本节已经长期记录摘要、失败集合和日志哈希。

#### workspace路径结论

48/48 external-workspace kid全部通过。每个 kid都通过：

- BF16 Y；
- FP32 Y；
- caller Tensor复用且指针不变；
- 自动 `torch.empty`恰好一次、shape/dtype/device正确；
- 同步后数值正确，自动 workspace Tensor没有被隐藏 Python/C++ registry持有。

因此 gfx950任务一新增的 Torch-owned split-K workspace路径已经覆盖全部登记 workspace kid，
不是只验证代表 kid 200。

#### 10个失败的窄分类

失败 kid精确为：

```text
1400, 1401, 1402, 1403, 1404
6400, 6401, 6402, 6403, 6404
```

它们全部属于 `a16w16_mono_tile`及4G-safe镜像，都是 non-workspace。测试循环先执行 BF16 Y，
其断言通过后才进入 FP32 Y；FP32结果约99.6%--99.7%元素不匹配，最大绝对误差约44--50。
没有 timeout、OOM、hang或 illegal access。由于 direct测试拦截 `torch.empty`且未触发，不能把
这些失败归因于 Torch workspace分配或生命周期路径。

### 原始基线复现与会话中断

下一步原计划在最初基线
`ca68b4f3501762c15c550cb920a5516e9710cf89`复现上述10个 mono FP32 case：

1. 第一次 archive/隔离构建的 host TU因找不到 `ck_tile/core.hpp`失败；原因是临时源码只含
   submodule gitlink且没有向构建传真实 CK include目录；
2. 第二次使用
   `CK_DIR=/root/workspace/0810/aiter/3rdparty/composable_kernel`和新的
   `/tmp/aiter-gfx950-baseline-jit2.2B1fiZ`重启构建；
3. 06:02:16旧会话最后一个事件是等待该构建 session返回，没有对应工具返回和最终回复；
4. build log最后修改于06:02:22，只完成 registry/生成前段，没有
   `BASELINE_BUILD_AND_SMOKE_OK`；
5. 06:04:20内核记录旧 Docker veth被移除并释放/evict两个旧 AMD queues。没有 OOM、GPU
   reset、page fault或 RAS证据；中断是容器整体退出，不是 pytest正常失败。

因此基线归因仍开放：不能判定10项是最初版本既有 mono FP32缺陷，还是当前端点的独立
non-workspace回归。

### 冻结时资源快照与任务边界

2026-08-11 06:31 UTC，physical GPU 0--7均为100% GFX activity，每张显存约
`285.6--287.0 GB`；占用 PID `802980--802987`在当前容器不可见。当前没有残留 pytest/JIT
测试进程，不终止未知外部任务。

任务一在切换任务二时的精确边界：

- 已闭环：CPU/metadata/codegen合同、gfx950 focused suite、gfx950全部48个 workspace kid；
- 待归因：10个 gfx950 mono-tile non-workspace FP32 case；
- 待执行：gfx950原始/当前性能 A/B；
- 待对应硬件：gfx942/gfx1250数值、graph、并发，尤其 gfx1250 fused；
- 当前代码和本全量测试文件均未提交，任务二必须保留 dirty工作树，不得把开放项改写为已通过。

以后恢复任务一时，先在一张空闲 gfx950上完成修正 `CK_DIR`后的基线构建，只跑上述10个
FP32 case；不需要先重跑已经闭环的48个 workspace kid。

## 2026-08-11：任务一续测（基线归因闭环，性能因外部占用暂停）

### 恢复时资源和隔离环境

08:17:26 UTC检查时，physical GPU 0--7均为 MI355X/gfx950，GFX activity为0、每卡只有约
283 MB驱动基础显存，`amd-smi process`没有检测到运行进程。于是按冻结检查点恢复任务一，
且没有 reset/checkout当前 dirty工作树。

新建目录：

```text
baseline source: /tmp/aiter-gfx950-baseline-src3.EXHfY5
baseline JIT:    /tmp/aiter-gfx950-baseline-jit3.DLB5t1
result logs:     /tmp/aiter-gfx950-continuation-results.aVTGDs
```

源码由
`git archive ca68b4f3501762c15c550cb920a5516e9710cf89`提取；
`aiter/ops/opus/gemm_op_a16w16.py`与 `git show`内容的 SHA-256均为
`0e8e88453e38e4ced0c2620a07593646a1fb07f222b00e936e42935590d89cec`。
新 JIT没有沿用旧 `/tmp/aiter-gfx950-baseline-jit2.2B1fiZ` 的半成品、lock或生成目录。

编译环境包括：

```text
HIP_VISIBLE_DEVICES=0
GPU_ARCHS=gfx950
AITER_JIT_DIR=/tmp/aiter-gfx950-baseline-jit3.DLB5t1
AITER_META_DIR=/tmp/aiter-gfx950-baseline-src3.EXHfY5
CK_DIR=/root/workspace/0810/aiter/3rdparty/composable_kernel
PYTHONPATH=/tmp/aiter-gfx950-baseline-src3.EXHfY5
```

sidecar复制自首轮 current全量模块，共142项，SHA-256为
`b43395710e4d99e2e4ed5807dc495a6312e435b056d5f475d088496ff830bdf7`。
第一次只调用 `get_module()` 时因尚未触发 decorated raw binding、JIT metadata没有注册而得到
`ModuleNotFoundError`；它不是产品编译失败。改为用 kid 1400 BF16烟测触发 raw binding后，
生成器报告 `|S|=142`、只含 gfx950，`module_deepgemm_opus`在约13.6秒内构建成功。烟测最大
绝对误差为 `0.1248779296875`并通过容差，最终日志含
`BASELINE_BUILD_AND_SMOKE_OK 2026-08-11T08:22:26Z`。

### 10个 mono-tile case的基线和当前对照

定向 harness覆盖：

```text
1400, 1401, 1402, 1403, 1404
6400, 6401, 6402, 6403, 6404
```

每项使用 `seed=0x950000+kid`、`M=B_M`、`N=B_N`、`K=2*B_K`、batch=1，先运行 BF16 Y，
再运行 FP32 Y；golden为 FP32 `torch.bmm`，容差与 exhaustive test一致。

| 端点 | BF16 | FP32 | FP32超差比例 | FP32最大绝对误差 |
|---|---:|---:|---:|---:|
| `ca68b4f...` | 10/10 | 0/10 | 49.49%--50.53% | 80.38--102.26 |
| 当前 dirty端点 | 10/10 | 0/10 | 49.49%--50.53% | 80.38--102.26 |

两个日志的20条 case JSON去掉 `BASELINE_CASE`/`CURRENT_CASE`前缀后，`diff`为空。进一步比较
fresh codegen产物，5个 tile乘普通/4G-safe乘BF16/FP32得到的20个 mono-tile device TU在
基线与当前之间全部 `cmp`相同。

首轮 exhaustive日志曾报告约99.6%--99.8%超差、最大绝对误差约42.93--77.67；本轮比例
不同，是因为首轮先同时创建 BF16/FP32两个 output Tensor，本轮则在每个 dtype launch前创建
对应 output。两种 allocator时序都失败，表明这个既有缺陷的错误形态会受地址/分配布局影响；
它不改变归因结论。

因此此前开放的 mono归因现在闭环：10项是 `ca68b4f...` 已有的 non-workspace
`a16w16_mono_tile` FP32缺陷，当前端点没有新增失败；它们与 external Torch workspace无关。

原始日志及 SHA-256：

```text
917a1210b029dcf11016e3529bef3c9ffc6ea21a9af0bb7ca1d18607598a1e7b  baseline_build.log
a18c1d1cabfd8b02f233e882b67a261f8bb0e215fee94efe84c78cc24099d966  baseline_numeric.log
20dfcff981def137c5ab9e437df57e26981477135722758f71a97506e06da3a3  current_numeric.log
```

文件均位于 `/tmp/aiter-gfx950-continuation-results.aVTGDs/`。

### 性能 A/B启动后被外部八卡任务污染

性能计划覆盖48个 workspace kid（`200--223`、`1200--1223`）和BF16/FP32 Y，共96项。
每项使用与全量验收相同的 exact tile、`K=32*B_K`、`splitK=2`和 seed；先做数值断言，再做
20次 warmup、9轮乘100次 raw launch，取每轮单次耗时的 median。基线使用预热后的原内部
workspace，当前使用复用的 caller FP32 Tensor；计划按 A--B--B--A执行，并追加 graph replay
以分离 host validator/launch和 device kernel时间。

A1结束、B1/B2执行期间，宿主新出现 PID `2109525--2109532`。随后每张卡有一个对应进程常驻
约55.8--56.7 GB，8张卡均约18%--19% VRAM；进程不在当前容器中，且 `evicted_time`持续增长。
这违反独占前提，也说明已有测量可能包含调度/queue eviction影响。因此：

- `perf_baseline_A1.log`、`perf_current_B1.log`、`perf_current_B2.log`全部保留用于审计，但明确
  标记为无效，不能形成性能结论；
- 不执行 A2，不在外部进程存在时继续 graph-replay或其他GPU benchmark；
- 不终止、不驱逐这些未知进程；
- 资源释放后复用已经成功构建的 baseline/current JIT，从 A1完整重跑。

三个无效日志的 SHA-256分别为：

```text
09362a74d83d00208e413f518f0c51e5fe584b21429ebff10019478f3f2c57da
6ea7444e478ecadfe3f95b305e33321522be35e7071e8680f3e8e1a14209ce26
f1b78db26d666a16f3c727de5593e9e749c557752cf4734cee70246e0ebf5dfe
```

截至08:29:57 UTC，8张卡的GFX activity回到0，但外部进程和约18%显存仍在，仍不满足独占。
08:32:59 UTC再次采样时，同一外部八卡任务已升至每卡72%--75%显存、17%--66% GFX
activity，确认它不是可忽略的短暂驻留。当前 gfx950剩余项只有有效的原始/当前性能 A/B；
gfx942/gfx1250实机边界保持不变。

## 2026-08-11：任务一最终续测（有效性能 A/B、mono FP32根因与修复）

本节晚于上面的08:30 UTC暂停记录，是任务一当前权威结论。续测过程中没有 reset、checkout
或清理现有 dirty工作树，也没有终止外部进程。有效 A/B完成后，修复只修改 gfx950
non-workspace mono-tile epilogue；workspace benchmark目标集合不包含这些 mono kid，因此该
局部修复不会改变 A/B所测的48个 workspace kernel。

### 资源重新满足独占条件

外部八卡任务退出后，physical GPU 0--7均回到0% GFX activity、0% VRAM；`amd-smi process`
报告8张卡均无运行进程。有效性能轮次期间，在 A1、B1、B2、A2之间重复检查，状态仍为
0%/0%且没有KFD PID。所有有效性能数据来自单一 physical GPU 0上的串行进程，不与四卡
correctness shard并发。

最终全部测试退出后的再次检查也是8张卡0%利用率、0%显存，`amd-smi process`无运行进程，
当前容器没有残留 pytest、benchmark或JIT进程。

### 新增可复现的 A/B benchmark

新增：

```text
op_tests/bench_opus_gfx950_workspace_ab.py
```

当前为195行，SHA-256：

```text
0a96d5d917cf2a970ee9c9e8d2c70f632f59cfdbf81e8161742cc00d091ce87d
```

测试集合为所有 gfx950 external-workspace kid：

```text
200--223, 1200--1223
```

每个 kid使用 exact `M=B_M`、`N=B_N`、`K=32*B_K`、batch=1、split-K=2和固定 seed，分别
验证 BF16/FP32 Y，共96个 `(kid, dtype)` case。每项先对 FP32 `torch.bmm` golden做数值断言，
再执行：

- eager/raw：20次 warmup，9轮，每轮100次 launch，以每轮单次微秒数的 median为该轮结果；
- graph：在独立 stream预热后 capture，再用相同20/9x100口径测 `graph.replay()`；
- baseline调用旧 `_opus_gemm_a16w16_tune_raw`，由C++内部 workspace路径提供buffer；
- current调用 `_opus_gemm_a16w16_launch_raw`，显式传入复用的 FP32 Torch workspace；
- baseline的内部 workspace按 stream索引，所以在 capture前先在 graph stream调用旧
  `opus_gemm_workspace_init()`并预热当前shape，避免把首次注册/扩容放进capture。

第一次 graph尝试没有完成该 capture-stream初始化，属于不完整试跑，未混入最终目录或统计。
最终按 `A1 -> B1 -> B2 -> A2`顺序运行；每轮都是新进程，并使用与端点匹配的源码和隔离JIT：

```text
baseline source: /tmp/aiter-gfx950-baseline-src3.EXHfY5
baseline JIT:    /tmp/aiter-gfx950-baseline-jit3.DLB5t1
current source:  /root/workspace/0810/aiter
current JIT:     /tmp/aiter-gfx950-current.NtJydE
valid results:   /tmp/aiter-gfx950-perf-valid.HMflXg
```

四轮均输出96条 `PERF_CASE`，四轮均 `96/96` 数值通过。统计方法对每个
`(kid, dtype)`先计算两轮 baseline median的平均与两轮 current median的平均，再对配对值求和；
不是把全部原始samples无配对混池。

#### eager/raw结果

| dtype | baseline配对总和 | current配对总和 | current变化 | 逐项方向 |
|---|---:|---:|---:|---:|
| BF16 | `744.220 us` | `814.705 us` | `+9.471%` | 44慢 / 4快 |
| FP32 | `745.925 us` | `815.144 us` | `+9.280%` | 44慢 / 4快 |
| 合计 | `1490.145 us` | `1629.849 us` | `+9.375%` | 88慢 / 8快 |

#### graph replay结果

| dtype | baseline配对总和 | current配对总和 | current变化 | 逐项方向 |
|---|---:|---:|---:|---:|
| BF16 | `655.427 us` | `554.114 us` | `-15.457%` | 48快 / 0慢 |
| FP32 | `655.368 us` | `553.783 us` | `-15.500%` | 48快 / 0慢 |
| 合计 | `1310.795 us` | `1107.898 us` | `-15.479%` | 96快 / 0慢 |

eager/raw包含两端各自的Python/pybind/raw launch边界，graph replay测量已capture的device graph；
二者测量边界不同，因此结论必须并列保留，不能挑一项改写成笼统的“整体性能提升”或“整体
性能回退”。本轮只报告实测差值，不从这组数据额外推断未单独profile的host/device子成分。

有效日志 SHA-256：

```text
68009ecf5a67be8e95993d207fa7b1431375658d93d7a2cfa1bf2eaae3907888  perf_baseline_A1.log
97f91ee42c9fdb960ac950513f3f2f75fb8d961b8ddfc72f5d3bdf3ee1f37c1b  perf_baseline_A2.log
f83d8cda3e4c7597af0c490e4c0ed1bb94e7ea77053e0e21088337841a85452d  perf_current_B1.log
283fb22c7825c81ca92cfbae2be11cc2a378d7a757f9cde37f191e449504e14f  perf_current_B2.log
```

08:30 UTC节中的三份受外部进程污染日志仍保留作审计，但没有参与上述任何数值。

### mono FP32根因：旧kernel没有写Y

原失败 kid：

```text
1400, 1401, 1402, 1403, 1404
6400, 6401, 6402, 6403, 6404
```

它们分别是普通 `a16w16_mono_tile`和4G-safe镜像，全部是 non-workspace。基线/current逐字
一致的失败已经证明它不是 Torch-owned split-K迁移回归；本轮继续从实际写回路径定位。

mono traits固定逻辑 `VEC_C=8`。因此：

```text
BF16: 8 elements * 2 bytes = 16 bytes
FP32: 8 elements * 4 bytes = 32 bytes
```

`csrc/include/opus/opus.hpp` 的 `gmem::_store<vec>()`只为1、2、4、8、16字节提供
`raw_buffer_store`分支，32字节没有分支。模板的vector-size static_assert仍成立，所以FP32
specialization可以成功编译，但函数体不会发出任何store指令。

将Y预填为 `12345.0`后分别运行旧 kid 1400和6400的FP32 raw launch，两者都得到：

```text
changed=0/49152
```

这证明旧kernel不是“约一半算错”，而是完全没有写Y。先前约49.5%、99.6%等不同 mismatch
比例来自 `torch.empty`复用地址后Y中残留的旧数据；allocator时序改变残留形态，所以超差比例
随分配顺序变化。该现象不再作为计算正确率解释。

上游 `/root/workspace/gcnasm/opus_gemm/bf16_gemm/` 的 mono模板只实例化 BF16输出。AITER扩展
FP32 specialization后还存在第二个dtype假设：每个8元素chunk的lane-half交换硬编码为两对
`u32`，刚好覆盖8个BF16；8个FP32需要四对 `u32`。因此只补store仍不足以保证FP32元素顺序。

### 修复设计与被否决的两个近似方案

最终逻辑保持 `VEC_C=8`，不改变MFMA结果布局或lane映射：

1. `u32_per_half = sizeof(D_C) * 4 / sizeof(u32_t)`；BF16为2，FP32为4；
2. 对每个逻辑8元素chunk，逐对调用 `v_permlane16_swap_b32`；
3. `c_store_vec = 16 / sizeof(D_C)`；BF16为8，FP32为4；
4. 仍按原 vec=8 cached layout取得每个逻辑issue的基址；BF16发一笔 `store<8>`，FP32将
   value切成 `[0,4)`、`[4,8)`，在基址 `+0`、`+4`分别发两笔 `store<4>`。

两个实验说明为什么最终实现不能简单写成一次 bulk `store<4>`：

- 保留 vec=8 cached layout却直接调用 bulk `store<4>`时，只正确覆盖一半Y，kid 1400为
  `changed=24576/49152`；requested vec与cached issue-space不兼容；
- 把layout cache也改成vec=4时，第二个issue坐标展开为元素 `+1`而不是物理 `+4`，两笔store
  重叠，只覆盖每8元素中的5个，kid 1400为 `changed=30720/49152`。

最终实现显式从正确的vec=8逻辑base加 `group_offset=0/4`，避开上述歧义。BF16走
`if constexpr`单组分支，继续调用原 `store<8>`路径。

修改：

| 文件 | 当前相对HEAD numstat | 内容 |
|---|---:|---|
| `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_mono_tile_gfx950.cuh` | `+52/-17` | dtype-aware lane swap、16-byte物理store拆分、修正文档 |
| `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_mono_tile_4g_safe_gfx950.cuh` | `+52/-17` | 4G-safe镜像同一修复 |
| `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh` | `+3/-2` | 说明逻辑vec=8与FP32物理vec=4，并修正旧API名注释 |
| `op_tests/test_opus_a16w16_gemm.py` | `+28/-0` | 对1400/6400预填FP32哨兵，回归完整物理写回与数值 |

两份pipeline顶部不再声称kernel body与上游“byte-for-byte identical”，而是明确上游只实例化
BF16、AITER epilogue额外支持BF16/FP32。

### 最终 fresh JIT、sentinel与 ISA证据

最终目录：

```text
/tmp/aiter-gfx950-mono-final.xAJYPc
```

只复制全量 sidecar到新的空JIT目录，再由 raw kid 1400 launch触发
`module_deepgemm_opus`完整生成、编译、链接和加载。最终 sidecar含142项，SHA-256：

```text
b43395710e4d99e2e4ed5807dc495a6312e435b056d5f475d088496ff830bdf7
```

最终 sentinel smoke：

| kid | variant | dtype | changed | max abs vs FP32 golden |
|---:|---|---|---:|---:|
| 1400 | ordinary mono | BF16 | `49152/49152` | `0.12487793` |
| 1400 | ordinary mono | FP32 | `49152/49152` | `7.6293945e-06` |
| 6400 | 4G-safe mono | BF16 | `49152/49152` | `0.12463379` |
| 6400 | 4G-safe mono | FP32 | `49152/49152` | `7.6293945e-06` |

从最终 kid 1400、tile `192x256x64`的code object提取 ISA：BF16有12笔
`buffer_store_dwordx4`和24条 `v_permlane16_swap_b32`；FP32有24笔store和48条swap。FP32
store成对出现，第二笔带 `offset:16`字节，例如：

```text
buffer_store_dwordx4 ... offen
buffer_store_dwordx4 ... offen offset:16
```

code object SHA-256：

```text
4c5a77bd42321bf04ff00c5bd489f150f884e963eef444f7589ddb361dd66812  FP32
d73b03551c3db83bebdcc44615f7edc585b0ba5c22ded57051ce6041008e709f  BF16
```

### 最终 correctness和回归矩阵

定向原失败集合：

```text
10 passed, 130 deselected, 2 warnings in 3.43s
```

完整140-kid sweep继续使用 canonical registry和四个稳定ordinal shard，physical GPU 4--7各
35项：

| shard | pytest摘要 | 失败 | log SHA-256 |
|---|---:|---:|---|
| 0 | `35 passed` | 0 | `659e756a8b2a0faaccefadf06e8445d70c8ea8d2f78c514db6678d50e1006e19` |
| 1 | `35 passed` | 0 | `bf180a49dee30933ea2ec3e094e023d8c6bed37d3da50047cd7855007769fe79` |
| 2 | `35 passed` | 0 | `3bed0587228cab5e61bce6d1c3fcd855b8f4cc5dc3c14d63da617465b66887e1` |
| 3 | `35 passed` | 0 | `db35884c985bcca8238a966769e892b699b7eef9f34d7dd591d094a1bc31cd1c` |
| 合计 | `140 passed` | 0 | — |

最终日志与JUnit XML：

```text
/tmp/aiter-gfx950-mono-final.xAJYPc/results/shard{0,1,2,3}.log
/tmp/aiter-gfx950-mono-final.xAJYPc/results/shard{0,1,2,3}.xml
```

focused gfx950 suite在 physical GPU 4、5运行：

```text
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py

166 passed, 14 skipped, 2 warnings in 4.24s
```

新增的1400/6400 poisoned-output回归单独运行结果为
`2 passed, 9 deselected, 2 warnings in 3.89s`。14个skip均为需要gfx942/gfx1250硬件的
条件case。CPU过滤集：

```text
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  -k 'not raw_cpp and not graph_capture and not two_streams'

151 passed, 18 deselected, 2 warnings in 3.51s
```

18个deselect是明确未执行的GPU raw、graph和two-stream用例，不是失败。另有：

- `op_tests/test_opus_gfx950_exhaustive.py`当前305行，SHA-256
  `4391c420c2eecd3e9b1d697d5bc8c412f5fafa94cb223d4827afdb6a7aa3e243`；
- benchmark与exhaustive test的 `python -m py_compile`通过；
- 全工作树 `git diff --check`通过；
- 没有修改或清理用户已有的其他dirty文件。

### 最终边界

gfx950任务一现在已经完成：48个workspace kid、92个non-workspace kid、BF16/FP32输出、
workspace生命周期、graph、focused回归、完整140-kid验收和有效性能 A/B均有实机记录。

本轮机器仍只有gfx950。gfx942/gfx1250（尤其 gfx1250 fused）的实机数值、graph、双stream、
并发和性能仍不能由交叉编译或gfx950结果替代；这些是外部硬件补验，不是gfx950任务一的遗留
失败。

## 2026-08-11：workspace launch 性能优化第 2/3 项（实验记录；实现已回退）

本节晚于前述任务一最终续测，完整保留第2/3项性能实验及其A/B依据；不覆盖mono FP32修复、
140/140和原始A/B历史。`prepared`、`最终prepared`和“现在”等措辞描述的是当时实验端点，
不是当前源码状态；当前状态见本节末尾“第2/3项回退”。用户当时要求先实现：

1. 首次workspace launch完整校验，重复合同走prepared路径；
2. 合并`has_workspace(kid)`与`workspace_dispatch(kid)`的两次表查询。

### 修改范围

实现集中在：

```text
csrc/opus_gemm/opus_gemm.cu
csrc/opus_gemm/gen_instances.py
csrc/opus_gemm/codegen/gen_instances_gfx950.py
csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh
csrc/opus_gemm/include/gfx942/opus_gemm_arch_gfx942.cuh
csrc/opus_gemm/include/gfx1250/opus_gemm_arch_gfx1250.cuh
```

回归增加在：

```text
op_tests/test_opus_interfaces.py
op_tests/test_opus_workspace.py
```

现有dirty工作树包含任务一/任务二的其他修改，因此相对HEAD整文件numstat不能代表本轮增量；
本轮没有reset、checkout、clean或覆盖用户已有改动。

### gfx950 checked/prevalidated双wrapper

gfx950 FlatMM split-K generator现在生成一个共享实现：

```cpp
template <bool Validate, typename D_C>
static void <kernel>_impl(...);

template <typename D_C>
void <kernel>(...) {
    <kernel>_impl<true, D_C>(...);
}

void <kernel>_prevalidated(...) {
    <kernel>_impl<false, fp32_t>(...);
}
```

checked wrapper仍执行Y dtype、正extent/K偶数、prefetch下限、overflow、workspace容量/device/
dtype/contiguous/alignment和required byte span检查。prevalidated wrapper只跳过首调用已经证明且
本次匹配器重新保证不变的检查；split-K down-clamp、kargs构造、当前stream和两个kernel launch
继续每次执行。bias validator故意不放入`Validate`分支，因而bias可换地址且每次仍验证
contiguous、dtype和`[N]`/`[batch,N]`形状。

### thread-local prepared合同

`opus_gemm.cu`只保存POD标量和function pointer，不保存Tensor对象、storage ownership或引用：

```text
kid, split_k, prevalidated function
workspace ptr, validated numel lower bound, device
batch, M, N, K
```

命中时重新要求：XQ/WQ/Y均3D；XQ/WQ为BF16；Y为BF16或FP32；M/N/K/batch与已校验launch
相同；workspace仍是同一原地址，当前numel不少于首次validated Tensor，dtype为FP32，device
与首次及XQ一致，并仍contiguous。原地址已经通过non-null和16-byte alignment，因此相同地址
无需重复这两项。XQ/WQ/Y指针不缓存，launcher读取本次参数；bias也不进入cache key并逐次验证。

第一版曾逐字段比较XQ/WQ/Y/workspace完整shape/stride/dtype/device/numel。host profile表明这组
约50次比较本身抵消了省下的validator，于是最终版本只匹配checked路径真正依赖的安全和
geometry不变量。short、错误dtype、同地址non-contiguous view、misaligned slice等都会miss并
回到checked wrapper，不会静默进入fast wrapper。

### 单次workspace dispatch

generated workspace row从：

```text
{ kid, checked_func }
```

扩为：

```text
{ kid, checked_func, prevalidated_func }
```

架构router一次lookup返回`OpusA16W16WorkspaceDispatch{checked, prevalidated}`，删除热路径先
membership probe、再重复binary search取pointer的流程。gfx950 row填两个函数；gfx942和
gfx1250 row的第三字段为`nullptr`，仍只运行checked路径。非workspace五参数表保持独立，没有
把两种ABI混表。

### codegen与正确性验证

新增静态/codegen断言：

- gfx950全部48个workspace row均含`_prevalidated`，且manifest有对应声明；
- gfx942全部8个、gfx1250全部1874个workspace row的第三字段均为`nullptr`；
- 代表kid 200 generated实现同时含`impl<true>`和`impl<false>`；
- dispatch kid集合及digest与B0 golden一致。

最终全量gfx950 JIT：

```text
/tmp/aiter-gfx950-prepared2.TYzqMO
```

先复制142-entry sidecar再`AITER_REBUILD=1` fresh生成/编译/链接；sidecar SHA-256：

```text
b43395710e4d99e2e4ed5807dc495a6312e435b056d5f475d088496ff830bdf7
```

GPU回归先连续两次使用同一合法workspace，再分别传：同地址short view、同地址错误dtype view、
同地址non-contiguous transpose和misaligned slice，四项均回退checked并以预期错误拒绝。另以
相同geometry、同一workspace但全新XQ/WQ/Y地址运行BF16和FP32 Y，二者均与FP32
`torch.bmm` golden一致，证明fast wrapper不使用旧数据指针。

最终focused命令覆盖dispatch/workspace/graph/a16w16/interfaces，结果：

```text
217 passed, 15 skipped, 2 warnings in 4.60s
```

15项skip是当前机器没有gfx942/gfx1250的条件用例，不是失败。`git diff --check`及相关
`py_compile`均通过。

fresh三架构目录：

```text
/tmp/aiter-prepared-multiarch.qyHiYr
```

`GPU_ARCHS='gfx942;gfx950;gfx1250'`生成32-kid默认subset。下列四项均通过
`hipcc -std=c++20 -O3 --offload-arch=<arch> -fsyntax-only`：

```text
all_instances_host_gfx942.cu
all_instances_host_gfx950.cu
all_instances_host_gfx1250.cu
opus_gemm.cu（multiarch headers，gfx950 target host parse）
```

warning仅为仓库既有`aiter_tensor.h`忽略`hipFree*`返回值、`opus.hpp` deduction-guide attribute
和`--hip-link` unused；没有本轮compile error。gfx942/gfx1250仍没有实机数值或性能结论。

### 性能分层诊断

#### 1. 隔离C++ prepared路径

先把Torch Tensor预转换为pybind `aiter_tensor_t`并复用对象，只测同一kid 200的pybind/C++
launch。修改前`/tmp/aiter-gfx950-mono-final.xAJYPc`为`5.514345 us`，最终prepared为
`5.145200 us`：

```text
delta = -6.694%
```

这证明prepared sibling实际命中，并且合并查询/跳过validator在C++层有约`0.369 us`收益。

#### 2. 第2/3项修改前后端到端ABBA

正常`_opus_gemm_a16w16_launch_raw`仍经过`torch.ops`和`develop=True`转换。以mono-final JIT为
A、最终prepared JIT为B，在独占physical GPU 4按`A1 -> B1 -> B2 -> A2`运行全部48个workspace
kid乘BF16/FP32 Y：

| 口径 | dtype | 修改前配对总和 | prepared配对总和 | 变化 | prepared方向 |
|---|---|---:|---:|---:|---:|
| eager/raw | BF16 | `800.190 us` | `799.866 us` | `-0.040%` | 21快 / 27慢 |
| eager/raw | FP32 | `800.738 us` | `801.755 us` | `+0.127%` | 20快 / 28慢 |
| eager/raw | 合计 | `1600.927 us` | `1601.621 us` | `+0.043%` | 41快 / 55慢 |
| graph | BF16 | `556.103 us` | `555.904 us` | `-0.036%` | 26快 / 22慢 |
| graph | FP32 | `555.737 us` | `555.406 us` | `-0.060%` | 25快 / 23慢 |
| graph | 合计 | `1111.840 us` | `1111.310 us` | `-0.048%` | 51快 / 45慢 |

四轮期间存在整体频率漂移，但ABBA配对后两种口径都小于`0.05%`；结论只能写为正常Torch raw
端到端无可测收益、也无可测回退，不能把隔离C++的`-6.694%`直接外推到公开调用边界。

日志和SHA-256：

```text
3eac53b64047b1f7973a4914458b7208095d815496db30e0cea5e951ca04c233  perf_before_P0A1.log
20af62fea6baf5d18e8032f9909eb7b8692c5f9fa84b8d9ef4fe6c88e532822d  perf_compact_P2B1.log
5dde9081a82a92436a5980969a6e1e16010ba31daf7084923a94d5fefecc6412  perf_compact_P2B2.log
290af9ec09156c102b8d67e6bdf4b7fd12153dcf71a245a747c8ef54066577d3  perf_before_P0A2.log
```

目录：`/tmp/aiter-gfx950-prepared-perf.ydkarB`。

#### 3. 最初internal-workspace baseline与最终prepared

再用`ca68b4f...`隔离baseline按相同ABBA与最终prepared比较：

| 口径 | baseline配对总和 | 最终prepared配对总和 | 变化 | 最终方向 |
|---|---:|---:|---:|---:|
| eager/raw | `1453.401 us` | `1603.264 us` | `+10.311%` | 86慢 / 10快 |
| graph replay | `1275.734 us` | `1111.721 us` | `-12.856%` | 96快 / 0慢 |

日志SHA-256：

```text
e6bfd26aeae3f753a5ad017c09a92346297ef92a930cd329a747119e006a5799  perf_final_baseline_FA1.log
4ae87000528684eb2db7f174653b9e830c753228b87d49b92a151c2c9743dbda  perf_final_compact_FB1.log
4f28899a27edf56114ae0b622bf7f45ae8c7f44d9af9e818f3a5a546e9908ec4  perf_final_compact_FB2.log
30f40a4d4e46075c7cc515897ddc0cdfe578e5be0d0eb85948f0f0210f939ac7  perf_final_baseline_FA2.log
```

eager与graph仍必须分开解释：graph只回放captured device work，不经过逐调用Python/pybind
转换；eager包含完整host边界。当前`compile_ops(..., develop=True)`每次把Torch XQ、WQ、Y和
workspace读取`data_ptr/numel/shape/stride/dtype/device`并构造pybind对象。相比旧内部workspace
入口，新增workspace Tensor也要做一次这种转换。该层约11 us的总开销淹没了C++ fast path省下
的约0.37 us，所以继续删除C++检查不能解决剩余约1.5--2 us差距；workspace物理dtype也不是
这段host开销的根因。

若继续优化，需要单独授权更侵入性的prepared/pointer ABI：例如让重复launch复用已经转换的
workspace metadata，或通过只传当前data pointer的专用边界避免每次重新构造第四个pybind
Tensor合同，同时保留Torch Tensor生命周期在Python调用域内。不能用全局Tensor cache、恢复
C++ allocator或牺牲graph/stream所有权来换取该收益。

### 2026-08-11 13:22 UTC：第2/3项回退

#### 决策

用户确认下一步更深的 pointer/prepared ABI不依赖当前第2/3项后，授权回退这两项。依据是：

- 隔离pybind/C++层虽从`5.514345 us`降到`5.145200 us`，即`-6.694%`；
- 正常Torch raw端到端却只有eager `+0.043%`、graph `-0.048%`，均在噪声内；
- 剩余host差距来自Torch/custom-op/pybind对第四个Tensor metadata的逐调用转换，继续保留
  thread-local validator cache或合并一次binary search不能解决该层开销；
- 新pointer/prepared ABI可以重新封装预验证launcher，但不需要把当前实验实现作为前置依赖。

性能日志、SHA-256和上面的完整A/B表继续保留；本次没有重跑性能，因为目标是恢复修改前的
checked-only执行边界，而不是产生第三个性能端点。

#### 精确回退边界

仅撤销第2/3项，未执行`reset`、`checkout`或`clean`，也未覆盖并行任务已经完成的修改：

- `csrc/opus_gemm/codegen/gen_instances_gfx950.py`恢复每个gfx950 workspace kernel只有一个
  `template <typename D_C>` checked launcher；删除`<kernel>_impl<Validate, D_C>`和
  `<kernel>_prevalidated`，Y/extent/K/prefetch/overflow/workspace容量、device、dtype、
  contiguous、alignment检查重新每次执行；
- `csrc/opus_gemm/gen_instances.py`恢复workspace row为`{kid, func}`，manifest不再声明
  `_prevalidated`；gfx942/gfx950/gfx1250统一使用同一两字段生成合同；
- 三个arch header保持`OpusA16W16WorkspaceKidEntry {kid, func}`；
- `csrc/opus_gemm/opus_gemm.cu`保持独立`has_workspace(kid)`membership probe和
  `workspace_dispatch(kid)`lookup，不再有thread-local prepared状态；
- 删除`op_tests/test_opus_interfaces.py`中只断言三字段row、prevalidated manifest和
  `Validate`双wrapper的实验测试；原Torch workspace迁移测试（包括名字中历史性的
  `prepared_step5`）保留；
- 保留gfx950 mono-tile FP32物理store修复、Torch-owned workspace、graph/stream/lifetime
  合同、gfx1250 fused family以及任务二的接口/dispatch重构。

源码机械扫描：

```text
rg -n "prevalidated|PreparedWorkspace|OpusA16W16WorkspaceDispatch|<bool Validate|workspace_try_dispatch" \
  csrc/opus_gemm op_tests --glob '!**/__pycache__/**'

# exit 1，空输出
```

相关Python文件`py_compile`通过；全工作树`git diff --check`为exit 0。

#### fresh gfx950 JIT与实机回归

独立目录：

```text
/tmp/aiter-gfx950-rollback23.4aK1Lp
```

环境为`HIP_VISIBLE_DEVICES=0`、`GPU_ARCHS=gfx950`、首次`AITER_REBUILD=1`。生成物确认：

- `opus_gemm_a16w16_kid_dispatch.h`的workspace row只有`{kid, func}`；
- gfx950 generated workspace `.cuh`只有`template <typename D_C>`和
  `opus_validate_workspace(...)`，没有`_prevalidated`或`if constexpr (Validate)`；
- `module_deepgemm_opus.so`完整编译、链接并加载成功。

首次接口/workspace集合：

```text
pytest -q op_tests/test_opus_interfaces.py op_tests/test_opus_workspace.py

88 passed, 13 skipped, 2 warnings in 19.68s
```

复用同一fresh `.so`运行完整focused集合：

```text
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py \
  op_tests/test_opus_interfaces.py

218 passed, 23 skipped, 2 warnings in 4.48s
```

gfx950实机覆盖包括：kid 200 BF16与FP32输出均匹配Torch golden；exact typed workspace；
one-element-short、错误dtype、non-contiguous、misaligned workspace拒绝；graph capture/replay、
双stream、生命周期和既有mono FP32回归。23个skip均是当前机器缺少gfx942/gfx1250硬件的条件
case，不是失败。

#### fresh三架构生成与host语法

独立目录：

```text
/tmp/aiter-multiarch-rollback23.CQJaRY
```

`GPU_ARCHS='gfx942;gfx950;gfx1250'`生成32-kid默认subset；workspace table size为gfx950=6、
gfx942=7、gfx1250=6。以下四项均通过`hipcc -std=c++20 -O3 -fsyntax-only`：

```text
all_instances_host_gfx942.cu   --offload-arch=gfx942
all_instances_host_gfx950.cu   --offload-arch=gfx950
all_instances_host_gfx1250.cu  --offload-arch=gfx1250
opus_gemm.cu（同时包含三架构router，--offload-arch=gfx950）
```

warning仅为仓库/工具链既有的`--hip-link` unused、CK warning-group、`aiter_tensor.h`忽略
`hipFree*`返回值和`opus.hpp` deduction-guide attribute；没有compile error。gfx942/gfx1250仍
只有生成/host语法结论，没有新增实机数值或性能结论。

#### 当前结论

第2/3项已从当前代码回退，原实验数据保留。下一步更深的pointer/prepared ABI可以从当前
checked-only基线独立开始；若新ABI需要预验证launcher，应在新的handle/pointer生命周期和
所有权合同内重新设计，不应直接恢复本次端到端无收益的thread-local cache。
