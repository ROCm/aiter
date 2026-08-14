# OPUS Split-K Workspace Torch 化：任务一最终修改架构

> 从第 4 节 Step 1 开始施工，严格按 Step 1 → Step 6 执行。
>
> 本文只包含任务一：把 gfx950、gfx942、gfx1250 现有 a16w16 two-stage split-K workspace 改为 Torch 分配。共享 workspace 基础层按 family-neutral 设计，但本次只注册和修改 a16w16。
>
> 基线：`ca68b4f3501762c15c550cb920a5516e9710cf89`；分支：`splitk_to_torch_2`。

## 1. 范围与完成结果

本任务完成后，调用链固定为：

```text
Python 确定最终 (arch, family=a16w16, kid, split_k)
  -> 按最终 kid 生成 WorkspacePlan
  -> torch.empty(shape, dtype=D_WS, device=XQ.device)
  -> raw pybind(XQ, WQ, Y, bias, workspace, kid, split_k)
  -> C++ 按 arch + kid 查 generated dispatch table
  -> split-K launcher 直接使用 workspace.data_ptr()
  -> main kernel 写 partials
  -> reduce kernel 只读同一地址并写 Y
```

需要删除：

- C++ `hipMalloc/hipFree/hipHostMalloc` workspace 分配。
- `SplitkWsRegistry`、per-stream owner、host/device handle 和 mirror sync。
- `opus_splitk_ws_handle` 及 kernel 端二级指针解引用。
- graph capture prewarm 和 capture-stream 猜测逻辑。
- C++ `opus_gemm_workspace_init()` 实现和 pybind。

需要保留：

- 三架构现有 kernel 数值算法、split-K clamp 和 reduce 算法。
- 现有非 split-K launcher ABI。
- Python `opus_gemm_workspace_init()` deprecated no-op 一个 release，避免下游立即报 `AttributeError`。

明确不做：

- 不做 OPUS 接口全面重构。
- 不修改当前 a8w8、a8w4 MoE，也不虚构当前不存在的 a4w4 kernel。
- 不做 BMM 或 compiled-manifest 重构。
- 不引入 gfx1250 fused split-K pipeline。
- 不删除 C++ heuristic；任务一只把它保留为迁移对拍基准。

### 1.1 当前 family 适用性

本任务判断一个 kernel 是否接入 Torch workspace，只看它是否具有“main kernel 写 global partial workspace，随后 reduce kernel读取”的能力，不根据 `aXwX` 名称或名字里是否含 `split` 判断。

| 当前 family/kernel | 是否属于本任务 | 处理 |
|---|---:|---|
| a16w16 two-stage split-K | 是 | 三架构全部切到 Torch workspace |
| a16w16 普通/非 split-K | 否 | 保持原 launcher ABI |
| a16w16 atomic accumulate | 否 | 没有 external partial workspace，不传 workspace |
| 当前 a8w8 | 否 | 普通、scale、blockscale-bpreshuffle 均未使用这套 external workspace |
| a4w4 | 否 | 当前 `csrc/opus_gemm` 中不存在 |
| a8w4 | 否 | 当前属于 OPUS MoE，是另一套接口和 kernel |

因此，本次不会给 a8w8/a4w4 预先增加无用参数；只把共享分配和校验能力设计成将来可以被其他 family adapter复用。

当前基线可以得出两个同时成立的结论：

1. 当前所有使用 external global workspace 的 OPUS GEMM kernel 都属于 a16w16 two-stage split-K，所以本任务没有漏掉现存 a8w8/a4w4 workspace消费者。
2. 不能反推“所有 a16w16 都需要 workspace”；普通、persistent、mono-tile 和 atomic-accumulate kernel的 workspace planner仍然返回 `None`。

按 kid band 判断能力在当前树已经会出错：gfx942 kid `11000` 是 `a8w8_blockscale_bpreshuffle`，与 a16w16 kid共同位于 `gfx942_kernels_list` 和 `[10000, 20000)` 数值段。由此，`arch + kid区间` 也不能替代完整的 `(arch, family, kid)` 查询与 `WorkspacePlan | None` 结果。

## 2. PR #4246 的参考结论

已实际对照 [ROCm/aiter PR #4246](https://github.com/ROCm/aiter/pull/4246)：

```text
PR head:                         25dd628119019d12d6207f388d15ab2d8b7c409d
PR base:                         ca68b4f3501762c15c550cb920a5516e9710cf89
Torch workspace ownership:      dc2f48903248f67385e5088d102733d30165a50e
删除 workspace Tensor lru_cache: a9ed164ec8e9e0c5f4497a7e459d40af4d849c96
按 kid 修正 dtype/shape、删 prewarm: ea093a7742ca427db3ffb12c0162e2da70fa7318
```

采用以下设计：

- workspace 由 Torch caching allocator 管理。
- 使用 kernel 实际写入类型的 typed Tensor，不使用 `uint8` byte buffer。
- workspace dtype、tile padding 和容量由最终 kid 决定。
- launcher 接收外部 Tensor，并把 `data_ptr()` 直接传给 main/reduce kernel。
- 三架构全部迁移完成后删除 prewarm。

PR #4246 不能整段照搬，原因是它的 external-workspace ABI 只改了 gfx1250：

- gfx1250 split-K 使用带 workspace 的 6 参数 launcher。
- gfx950、gfx942 仍使用旧 5 参数 launcher和 `SplitkWsRegistry`。
- `SPLITK_REDUCE_ABI_MAP` 也只有 gfx1250 改成 direct pointer。

PR 自身还直接证明了两个安全要求：

- `dc2f4890` 按 kid `[20000, 30000)` 硬编码 `torch.empty(dtype=bf16)`；`ea093a77` 随后说明 fp32-workspace kid 覆盖了这个 bf16 buffer，导致相邻显存损坏并挂死机器。因此 dtype、shape和容量必须取自最终 kid，且 C++ 必须再次检查 dtype与容量。
- PR 最终 head 的 generic `opus_gemm()` gfx1250 heuristic路径仍临时执行 `hipMalloc`，并使用硬编码 `16 * padded_M * padded_N * sizeof(bf16_t)`。这说明 PR 只完成了显式/tuned kid路径；本任务必须先完成 Python selector，再关闭 generic bf16 launch，才能真正删除所有生产路径 allocator。

本任务在它的设计上补齐三架构，同时不采用：

- 缓存 workspace Tensor 的 `functools.cache/lru_cache`。
- OPUS 模块内部修改 `sys.path`。
- 实例数据加载失败后猜一个最大 buffer 继续 launch。
- 用 kid 数值区间判断 workspace 能力。
- generic C++ 路径临时 `hipMalloc` workspace。
- PR 中的 fused split-K、TDM 重构和调优数据变更。

## 3. 最终 ABI 与数据契约

### 3.1 共享基础层与 family adapter

共享层只负责 workspace 生命周期，不理解 a16w16 的 tile、scale 或 heuristic：

```text
KernelKey(arch, family, kid)
WorkspacePlan | None
allocate_workspace(plan, device)
validate_workspace(tensor, plan)
checked extent/capacity helpers
```

family adapter负责各自的选择和 ABI：

```text
selector
workspace plan公式
launcher函数指针类型
generated kid dispatch table
main/reduce kernel参数
```

本次只实现 `a16w16 adapter`。不能把所有 family 强行塞进一个函数指针类型：例如未来 a8w8 workspace launcher 还可能需要 `x_scale/w_scale`，a4w4 还可能需要 packed-weight metadata，它们必须拥有各自的 launcher ABI 和 dispatch table。

可扩展性约束：

- 共享 `_workspace.py` 中不出现 a16w16 kid区间、tile 常量或 arch 特判。
- family planner返回 `WorkspacePlan | None`；即使名字含 `splitK`，atomic/fused kernel不使用 external workspace时仍返回 `None`。
- C++ 通用 checked-multiply、device/contiguous/alignment/capacity 校验放在共享 helper；dtype、shape 和 launcher参数由 family adapter提供。
- 将来增加新 family 时新增其按需查询/planner/dispatch，不修改现有 a16w16 表，也不改共享 allocator。

### 3.2 按需读取现有实例数据

`csrc/opus_gemm/opus_gemm_common.py` 继续是唯一实例数据源，但不新增 12 字段 `KernelSpec`、frozen catalog或 `_catalog.py` facade。

运行时先导入 `aiter.jit.core`（`gemm_op_a16w16.py` 已经为了 `compile_ops` 这样做）；它会把 `AITER_META_DIR` 集中加入 `sys.path`。随后直接使用仓内既有模式：

```python
from csrc.opus_gemm.opus_gemm_common import ...
```

禁止在 `aiter/ops/opus` 内再次修改 `sys.path`，也不为一次直接 import 增加纯转发模块。

任务一直接复用 `OpusGemmInstance` 已有字段：

```text
kernel_tag, name/launcher_symbol
B_M, B_N, B_K
arch_prefix
splitk_workspace_dtype
```

其中 `splitk_workspace_dtype` 已存在：默认 fp32，gfx942 bf16ws实例已经覆写为 bf16，本任务只读取，不重复新增。

只补任务一实际需要的窄查询/常量：

```text
get_kernel_instance(arch, family, kid) -> OpusGemmInstance | None
kernel_needs_external_workspace(arch, family, kid) -> bool
GFX942_BF16WS_EXACT_N = {64, 128, 256, 384, 512, 1024, 2048}
```

split-K上限、auto/clamp和shape合法性继续由已有函数计算，不冻结成新 metadata字段。gfx1250 `batch==1` 作为 a16w16 planner/launcher约束实现，不扩充全局实例 schema。

规则：

- kid 身份必须是 `(arch, family, kid)`，不能只查裸整数。
- 所有查询必须显式指定 `family="a16w16"`，避免误收 gfx942 a8w8 kid。
- 是否需要 workspace 由窄查询与最终 `WorkspacePlan | None` 决定，不能用 family 名、kernel tag字符串或 kid数值区间推断。
- 实例数据无法加载或 kid 未找到时明确失败，不能猜 workspace。
- gfx1250 当前 two-stage kernel 是 `single_only`；`batch != 1` 必须在 Python 和 C++ 两层拒绝。

gfx942 bf16ws 必须在 Python 分配前解析成真正会执行的 launcher：

| requested kid | N 不在 exact-N 集合时的当前 C++ 行为 | Python 最终行为 |
|---|---|---|
| 10210（kbuf1 legacy bf16ws） | 静默转跳 10200 fp32ws | 分配前把 actual kid解析为10200 |
| 10213（kbuf2v bk128 bf16ws） | 静默转跳 10203 fp32ws | 分配前把 actual kid解析为10203 |
| 10216（quad MFMA32 bf16ws） | `AITER_CHECK` 失败，不转跳 | 分配前直接拒绝 |

exact-N集合必须由 codegen、Python selector和tuner共享同一常量。当前 codegen包含 `N=384`，而 tune侧旧副本遗漏了384；任务一顺手消除这两个副本。

最终 `LaunchConfig` 同时区分 requested kid和actual kid，WorkspacePlan只能读取 actual kid。generated launcher中的 host redirect随后删除，禁止在拿到 bf16 workspace后再换成fp32-workspace launcher。

### 3.3 a16w16 两套 launcher table

非 split-K launcher 不需要 workspace，保持现有签名：

```cpp
using OpusA16W16Kernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    std::optional<aiter_tensor_t> bias, int split_k);
```

三架构 split-K launcher 统一为独立签名：

```cpp
using OpusA16W16WorkspaceKernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    aiter_tensor_t& workspace,
    std::optional<aiter_tensor_t> bias, int split_k);
```

codegen 生成两张 kid 表：

```text
non-workspace kid -> OpusA16W16Kernel
workspace kid     -> OpusA16W16WorkspaceKernel
```

约束：

- 原非 split-K 表和所有非 split-K launcher 签名不做机械加参。
- split-K 表按 arch 分区并覆盖 gfx950、gfx942、gfx1250，不只是 gfx1250；错误 arch 的 kid 不能命中当前设备的表。
- C++ 根据当前 arch 在 generated workspace table 中查 kid；命中才要求 workspace。不能用 `[200, 300)` 一类区间判断。
- `opus_gemm_a16w16_tune()` 的 raw ABI 显式接收无默认值的 optional workspace：

```cpp
void opus_gemm_a16w16_tune(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    std::optional<aiter_tensor_t> bias,
    std::optional<aiter_tensor_t> workspace,
    int kernel_id,
    int split_k);
```

- Python 公共 wrapper 保持旧参数顺序，并把 `workspace=None` 作为末位 keyword-only 高级参数，避免破坏旧的 positional `(kid, splitK)` 调用。
- non-workspace kid 必须传 `None`；workspace kid 必须传 Tensor。
- generic `opus_gemm()` 的 bf16 生产路径必须停用，正常 bf16 调用统一先经过 Python planner；否则它仍可能在没有 workspace 的情况下选到 split-K kid。generic a8w8 分支不改。

这里的函数指针类型只属于 a16w16 adapter，不作为未来 a8w8/a4w4 的通用 ABI。

### 3.4 kernel 指针 ABI

main kernel 会写 workspace：

```cpp
void* ptr_ws;
```

reduce kernel只读 workspace：

```cpp
const void* ws_ptr;
```

traits 中明确 workspace 类型 `D_WS`；累加类型 `D_ACC` 和最终输出 `D_OUT` 不再被拿来计算 workspace 宽度。所有 workspace cast、`sizeof` 和容量计算只引用 `D_WS`。

gfx942 还有一项不能机械删除的实现语义：当前 `opus_splitk_ws_ptr()` 不仅解引用 `ws_handle->ptr`，还把64位地址拆成高低32位并通过 `__builtin_amdgcn_readfirstlane` 在 wave 内统一。direct-pointer迁移会删除前者，但本次先保留后者：用新的 `opus_gfx942_uniform_ws_ptr(ptr_ws)` 取代旧 helper，main/reduce pipeline继续通过它取得 `D_WS*`。这样旧 `opus_splitk_ws_*` 命名和handle依赖可以完全消失。只有在反汇编确认直接读取kernel参数已经生成等价的scalar/uniform地址路径，并完成寄存器与性能对比后，才可在后续优化中删除 `readfirstlane`；不能在workspace所有权切换中顺手假设它无用。

### 3.5 a16w16 WorkspacePlan

```text
padded_M = ceil_div(M, B_M) * B_M
padded_N = ceil_div(N, B_N) * B_N
```

| arch | typed workspace shape | dtype | batch 规则 |
|---|---|---|---|
| gfx950 | `[allocation_split_k, batch, padded_M, padded_N]` | fp32 | 支持 batch |
| gfx942 | `[allocation_split_k, batch, padded_M, padded_N]` | 由 kid 决定 bf16/fp32 | 支持 batch |
| gfx1250 | `[allocation_split_k, padded_M, padded_N]` | fp32 | 只允许 batch=1 |

`allocation_split_k` 使用 Python 已解析的 requested/auto 初值；launcher只会向下 clamp，因此分配是安全上界。最大值由已有 per-kid split-K规则计算，荒谬大的 split-K 在 `torch.empty` 前拒绝。

generated launcher继续按 clamp 后的 `effective_split_k` 计算精确 `required_numel`，作为防越界的最后一道检查。

## 4. 施工顺序

### 施工前基线门禁

本文只对基线 `ca68b4f3501762c15c550cb920a5516e9710cf89` 的现存源码负责。开始改代码前重新执行：

```bash
git rev-parse HEAD
git status --short
git ls-remote https://github.com/ROCm/aiter.git \
  refs/pull/4246/head refs/pull/4320/head
```

如果基线在施工前发生变化：

- PR #4246 若先合入，除本文清单外还要删除它新增的 `opus_gemm_workspace_release()` / `opus_gemm_workspace_release_all()` 及 Python/pybind导出，并重新确认 generic heuristic `hipMalloc` 已被覆盖。
- PR #4320 若先合入，先暂停并重新枚举 workspace消费者。它新增 a8w8 MX-scale/BMM split-K 路径及额外 raw allocation，不属于本文当前 a16w16基线，不能悄悄套用 a16w16 launcher ABI，也不能继续宣称“当前所有 workspace消费者均已覆盖”。
- 任何新 family 都先按 `(arch, family, kid)` 查询并生成 `WorkspacePlan | None`，再决定是扩展本任务还是另开交付。

Step 1 是独立的无行为变化提交。Step 2–Step 5 共同构成 workspace ABI 切换，必须作为一个原子提交：中间可以按顺序编辑，但不能在 Step 5 接通新入口前删除旧 allocator/prewarm，也不能把中间态当作可运行版本。

### Step 1：让 Python 在分配前知道最终 kid 和 split-K

先做这一项，不改 workspace ABI。

这一步不可省略：#4246 最终 head 的 tuned/explicit路径能由 Python 按 kid 分配，但 generic heuristic路径仍在 C++ 临时 `hipMalloc`。只有 Python 先解析出最终 kid/split-K，正常生产调用才都能经过同一个 WorkspacePlan。

修改：

```text
csrc/opus_gemm/opus_gemm_common.py
csrc/opus_gemm/opus_gemm_tune.py
aiter/ops/opus/gemm_op_a16w16.py
```

新增：

```text
aiter/ops/opus/_selector_a16w16.py
aiter/ops/opus/heuristics/__init__.py
aiter/ops/opus/heuristics/a16w16_gfx1250.py
aiter/ops/opus/heuristics/a16w16_gfx950.py
aiter/ops/opus/heuristics/a16w16_gfx942.py
op_tests/test_opus_dispatch.py
```

动作：

1. `gemm_op_a16w16.py` 在导入 `aiter.jit.core` 后直接从 `csrc.opus_gemm.opus_gemm_common` 读取现有实例和窄查询；不新增 `_catalog.py`，不修改局部 `sys.path`。
2. `opus_gemm_common.py` 只增加第 3.2 节列出的窄查询、gfx942 exact-N共享常量和必要的family过滤；不生成 frozen metadata投影。
3. `opus_gemm_tune.py` 删除自己的 `BF16WS_EXACT_REDUCE_SHAPES` N集合副本，改为消费共享常量；保留它确实需要的 row-count/tune规则。
4. 依次等价移植 gfx1250、gfx950、gfx942 heuristic；gfx942 用现有 instance `.name` 做 `launcher_symbol -> kid` 反查，不手写第二张映射。
5. 单独移植 gfx942 auto split-K 和向下 clamp 规则；旧 gfx942 selector不返回 split-K，因此 kid parity 与 split resolver 分开对拍。
6. `_selector_a16w16.py` 固定执行 `explicit -> tuned CSV -> Python heuristic -> framework fallback`；它是 a16w16 adapter，不作为所有 family 的统一 selector。
7. 在 selector末端解析 requested/actual kid：10210/10213 的非法N分别改选10200/10203，10216非法N直接拒绝；随后才生成 WorkspacePlan。
8. tuned kid 不合法时原子丢弃 `(kid, splitK)`，不能只换 kid 后沿用旧 split-K；上述历史 bf16ws redirect是显式的 actual-launcher解析，不属于沿用失效 tuned row。
9. `gemm_op_a16w16.py` 的 CSV miss 不再调用 generic C++ bf16 selector；正常路径统一拿到最终 `LaunchConfig` 后调用 tune wrapper。
10. C++ heuristic/lookup 暂不删除，只作为 parity golden probe。

Step 1 验收：三架构 CSV-miss shape corpus 的 kid 与旧 C++ 完全一致；gfx942 effective split-K 单独完全一致。通过后单独提交。

### Step 2：增加 WorkspacePlan 和集中 Torch 分配

新增：

```text
aiter/ops/opus/_workspace.py
aiter/ops/opus/_workspace_a16w16.py
op_tests/test_opus_workspace.py
```

修改：

```text
aiter/ops/opus/gemm_op_a16w16.py
```

动作：

1. `_workspace.py` 只实现动态 `WorkspacePlan`、typed Tensor分配和通用验证，不包含 a16w16/arch/kid 特判。
2. `_workspace_a16w16.py` 根据 actual `OpusGemmInstance + shape + split_k` 生成第 3.5 节的 a16w16 plan或 `None`。
3. 在 `gemm_op_a16w16.py` 准备集中执行的 `resolve -> plan -> torch.empty -> raw launch` 路径；Step 5 raw ABI 接通后再启用，不得缓存 workspace Tensor。
4. 显式传入 workspace 时复用共享验证逻辑，不再分配。

### Step 3：把 codegen 拆成 non-workspace / workspace 两套 dispatch

修改：

```text
csrc/opus_gemm/gen_instances.py
csrc/opus_gemm/codegen/gen_instances_gfx950.py
csrc/opus_gemm/codegen/gen_instances_gfx942.py
csrc/opus_gemm/codegen/gen_instances_gfx1250.py
csrc/opus_gemm/include/opus_gemm_common.cuh
csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh
csrc/opus_gemm/include/gfx942/opus_gemm_arch_gfx942.cuh
csrc/opus_gemm/include/gfx1250/opus_gemm_arch_gfx1250.cuh
```

`gen_instances.py`：

1. `SPLITK_REDUCE_ABI_MAP` 三个 arch 全部改为 `const void* ws_ptr`。
2. 增加 split-K host signature；manifest 对 split-K 声明 6 参数，对非 split-K 保持 5 参数。
3. `gen_a16w16_tune_lookup()` 生成 non-workspace 表和 workspace 表，不能把两种函数指针放进同一表。
4. `(M,N,K)` runtime lookup 不再生成可直接 launch 的 split-K 5 参数函数指针；旧 selector probe只返回 kid，不负责 launch。
5. subset-compile 的 CSV kid 集合和 `HEURISTIC_DEFAULT_KIDS` 保留，防止 Python selector选到未编译 kid。

`opus_gemm_common.cuh` 只增加 family-neutral 的 checked extent和基础 workspace校验 helper；a16w16函数指针类型及 dispatch仍留在 a16w16 arch/dispatch层。

三个 arch codegen：

1. 所有 split-K launcher 增加 `aiter_tensor_t& workspace`。
2. 删除 capture 检测、registry get/grow、raw allocation、handle mirror 和 sync。
3. 从 `workspace.data_ptr()` 取得 direct pointer。
4. 保留原 split-K clamp、shape 物理约束、main launch 和 reduce launch。
5. clamp 后用 checked `size_t` 计算 `required_numel`并检查容量。
6. gfx942 对 10210/10213 删除静默 fp32ws转跳，并为全部 bf16ws launcher保留/增加 exact-N `AITER_CHECK` 作为 raw C++ 最后防线；10216继续硬失败。正常 Python路径必须已在 Step 1 解析actual kid。
7. `gen_instances_gfx942.py` 从共享 `GFX942_BF16WS_EXACT_N` 生成host条件，并断言 device exact-rowblock配置导出的N集合与其一致。

三个 arch dispatch header：

1. 现有 non-workspace dispatch 只查 non-workspace 表。
2. 新增 workspace dispatch，返回 `OpusA16W16WorkspaceKernel`。
3. arch router 以 generated workspace table membership 判断 kid 类型，不维护数值区间副本。

### Step 4：把三架构 kernel 改成 direct pointer

修改 traits：

```text
csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh
csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_traits_a16w16.cuh
csrc/opus_gemm/include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh
```

动作：删除三份 `opus_splitk_ws_handle` 和 guard；kargs 改为 `void* ptr_ws`；workspace 类型统一由 `D_WS` 表达。gfx942 删除旧 `opus_splitk_ws_ptr()` 符号，用接收direct `ptr_ws`的 `opus_gfx942_uniform_ws_ptr()` 保留现有64位 `readfirstlane` uniform化；新helper不读取handle或device mirror。

修改 main pipeline：

```text
csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh
csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_em3en4_lds1_pgr2_sk.cuh
csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_kbuf1.cuh
csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_kbuf2v.cuh
csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_kbuf2v_bk128.cuh
csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_quad_mfma32_kbuf1.cuh
csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh
csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh
```

gfx950/gfx1250把 `ws_handle->ptr` 改为 `reinterpret_cast<D_WS*>(kargs.ptr_ws)`。gfx942的main pipeline改成 `opus_gfx942_uniform_ws_ptr<D_WS>(kargs.ptr_ws)`，复用上一段的direct-pointer uniform helper；其余地址公式不变。

修改 reduce：

```text
csrc/opus_gemm/include/gfx950/splitk_reduce_gfx950.cuh
csrc/opus_gemm/include/gfx942/a16w16/splitk_reduce_gfx942.cuh
csrc/opus_gemm/include/gfx1250/splitk_reduce_gfx1250.cuh
```

统一接收 `const void* ws_ptr`。gfx950/gfx1250直接cast为 `const D_WS*`；gfx942 reduce通过direct-pointer uniform helper取得 `const D_WS*`。gfx942 必须覆盖 baseline、bf16ws、exact-N/OOB overload 和全部转发层。

### Step 5：修改 raw entry/binding，最后删除旧 allocator

修改：

```text
csrc/opus_gemm/include/opus_gemm.h
csrc/include/rocm_ops.hpp
csrc/pybind/opus_gemm_pybind.cu
csrc/opus_gemm/opus_gemm.cu
aiter/ops/opus/gemm_op_a16w16.py
aiter/ops/opus/__init__.py
aiter/tuned_gemm.py
```

动作：

1. 按第 3.3 节给 raw tune entry 增加 optional workspace。
2. workspace table命中时验证 Tensor并调用 6 参数 launcher；否则要求 `workspace=None` 后调用原 5 参数 launcher。
3. 启用 `gemm_op_a16w16.py` 在 Step 2 准备的 Torch workspace 路径。
4. 停用 generic `opus_gemm()` bf16 launch；a8w8 分支保持原样。
5. 删除 `SplitkWsRegistry`、三个 `opus_splitk_ws_*` 函数、`opus_gemm_workspace_init()` C++ 实现和相关 `<mutex>/<unordered_map>` include。
6. 删除 `OPUS_GEMM_WORKSPACE_INIT_PYBIND` 及其调用。
7. `aiter/tuned_gemm.py` 删除 `_opus_needs_ws_prewarm`、capture stream 猜测、prewarm set 和调用点。
8. `opus_gemm_workspace_init()` 改为 Python deprecated no-op。

下列调用方继续调用公共 wrapper，由 wrapper 自动分配；只需回归验证，不另写 workspace 分配逻辑：

```text
csrc/opus_gemm/opus_gemm_tune.py
csrc/gemm_a16w16/gemm_a16w16_tune.py
aiter/ops/deepgemm.py
```

C++ 对 workspace kid 必须检查：

```text
workspace 存在
workspace.device_id == XQ.device_id
workspace.dtype == D_WS
workspace.is_contiguous()
workspace.data_ptr() 满足该 kid 的对齐要求
checked required_numel <= workspace.numel()
```

所有 extent 乘法先做 `size_t` overflow check。只检查 `numel` 不够：dtype 错误时元素数可能相同、实际字节数却不足。

### Step 6：测试与文档收尾

新增或扩展：

```text
op_tests/test_opus_dispatch.py
op_tests/test_opus_workspace.py
op_tests/test_opus_graph.py
op_tests/test_opus_a16w16_gemm.py
```

更新：

```text
csrc/opus_gemm/README.md
aiter/ops/opus/README.md
```

必须覆盖：

| 类别 | 验收内容 |
|---|---|
| selector parity | 三架构 Python kid 与旧 C++ 一致；gfx942 split resolver单独一致 |
| 数值 | 三架构现有 a16w16 split-K 与 torch golden 一致 |
| dtype | gfx942 bf16/fp32 workspace；gfx950/gfx1250 fp32 workspace |
| 容量 | 正好容量成功，少 1 element 失败 |
| 负例 | 缺失、错 dtype、错 device、noncontiguous、错 alignment 均失败 |
| gfx942 redirect | 10210→10200、10213→10203、10216拒绝；覆盖 exact-N集合内外并单测 N=384 |
| gfx942 uniform pointer | 对比迁移前后ISA/寄存器与split-K性能，确认direct pointer仍保留等价的wave-uniform/readfirstlane语义 |
| gfx1250 batch | batch>1 在 Python 和 C++ 两层失败 |
| bias回归 | gfx1250覆盖`Y=bf16 + bias=fp32`；gfx950/gfx942保持当前各自支持/拒绝及dtype规则，不被共享workspace validator改变 |
| graph | capture/replay 多次与 golden 一致，无 prewarm |
| 并发 | 双 stream/TBO 各自持有调用级 workspace，不共享 scratch |
| 生命周期 | 多 shape 后无 Python 全局 Tensor cache 导致的持续显存常驻 |
| split-K 上限 | 超上限在 `torch.empty` 前失败 |
| scope isolation | 现有 a8w8、a8w4 MoE API/launcher/codegen产物不因本任务改变 |

任务一最终完成定义：

- gfx950、gfx942、gfx1250 的现有 a16w16 two-stage split-K 全部使用调用级 typed Torch workspace。
- Python 在分配前已经确定 requested/actual `(arch, family, kid, split_k)`，且 workspace由 actual instance生成。
- C++ a16w16范围内不存在 raw workspace allocator、registry、handle、mirror或prewarm。
- gfx942 direct-pointer路径已保留或用ISA证明确认等价的wave-uniform地址语义，没有因删除handle helper引入寄存器/性能回退。
- 共享 workspace层没有 a16w16 kid区间、arch 特判或 launcher参数；a16w16 selector/planner/dispatch仍明确属于 family adapter。
- a8w8、a8w4 MoE 和不存在的 a4w4 不被本任务修改；未来只有出现 external two-stage workspace kernel时才新增对应 adapter。

最终机械检查：

```bash
rg -n "SplitkWsRegistry|opus_splitk_ws_|ws_handle" \
  csrc/opus_gemm aiter/ops/opus aiter/tuned_gemm.py

rg -n "hipMalloc|hipFree|hipHostMalloc" \
  csrc/opus_gemm/opus_gemm.cu \
  csrc/opus_gemm/codegen/gen_instances_gfx950.py \
  csrc/opus_gemm/codegen/gen_instances_gfx942.py \
  csrc/opus_gemm/codegen/gen_instances_gfx1250.py

rg -n "sys\.path\.(insert|append)" aiter/ops/opus csrc/opus_gemm
git diff --check
```

允许的例外只有 Python deprecated `opus_gemm_workspace_init()` 空壳和 test-only legacy selector probe；任何 workspace handle、raw allocator、prewarm 或按错误类型计算 workspace 宽度的命中都必须消失。
