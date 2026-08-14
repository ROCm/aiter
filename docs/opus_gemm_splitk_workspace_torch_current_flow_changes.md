# OPUS split-K workspace：最初版本与当前 Torch 管理版本直接对比

更新时间：`2026-08-11`

本文只做一个端点对端点的比较：

- **最初版本**：提交 `ca68b4f3501762c15c550cb920a5516e9710cf89`，workspace 完全由
  OPUS/C++ launcher 隐式管理，Python 和 raw tune ABI 都没有 workspace Tensor；
- **当前版本**：分支 `splitk_to_torch_2`、HEAD
  `2352c46c784d6ba3a0c71ff89b4bdb4c2fefa59f` 之上的当前工作树，workspace 是普通的调用级
  Torch Tensor，由最终 `actual_kid` 决定是否分配、shape 和 dtype。

本文不再把 HEAD 中曾存在的 `WorkspacePlan` 方案称为“原流程”。该方案只是迁移过程中的中间
步骤：最初版本没有这两个模块，当前版本也已经删除它们。

## 1. 一页看懂：真正的原流程与当前流程

### 1.1 最初版本：C++ launcher 隐式拥有 workspace

```text
gemm_a16w16_opus(A, B, ...)
  |
  |-- 显式 kernelId
  |     -> opus_gemm_a16w16_tune(..., kernelId, splitK)
  |     -> raw ABI 不含 workspace Tensor
  |
  |-- tuned CSV 命中
  |     -> opus_gemm_a16w16_tune(..., solidx, splitK)
  |     -> raw ABI 不含 workspace Tensor
  |
  `-- tuned CSV 未命中
        -> _opus_gemm_bf16_dispatch(...)
        -> C++ opus_gemm(...)
        -> C++ tuned lookup / architecture heuristic

进入 generated split-K launcher 后：
  -> 用 kid 数值区间判断是不是 split-K
  -> 取得当前 hipStream_t
  -> 查询进程级 mutex registry（key = hipStream_t）
  -> 得到 host-pinned workspace handle及可选 device mirror
  -> launcher按 M/N/K/batch/split-K计算所需 bytes
  -> 容量不足时 launcher内部 hipMalloc；旧 buffer用 hipFree释放
  -> main/reduce kernel通过 handle取得 workspace地址
  -> main kernel写 partial，standalone reduce kernel读 partial并写 Y
```

最初版本的 Python 调用者看不到 workspace，也不能传入、复用或检查一个 workspace Tensor。
workspace 的地址、容量、stream 归属和扩容生命周期都隐藏在 C++ 内部。

### 1.2 当前版本：最终 actual kid 驱动调用级 Torch Tensor

```text
gemm_a16w16_opus(A, B, ...)
  -> select_launch_config(...)
       1. explicit override
       2. tuned CSV
       3. Python architecture heuristic
       4. framework fallback
       得到唯一 resolved LaunchConfig / actual_kid

  -> framework fallback ?
       是：torch.bmm路径
       否：_init_a16w16_workspace(config, XQ, Y, workspace=None)
             -> 用 (arch, family, actual_kid) 查询 exact instance
             -> 用 SPLITK_KIDS membership判断是否需要 external workspace
             -> 从 exact kid读取 tile、workspace dtype和 fused SplitK
             -> 计算 two-stage或 fused shape
             -> 自动 torch.empty(...)，或原样复用 caller Tensor

  -> raw ABI(XQ, WQ, Y, bias, workspace, actual_kid, launch_split_k)
  -> C++按 generated membership分到 non-workspace或workspace表
  -> generated launcher校验 device/dtype/contiguous/alignment/capacity
  -> workspace.data_ptr()直接传给 kernel
  -> two-stage main + standalone reduce
     或 fused clustered main + 同 kernel内 reduce
```

当前唯一真值链是：

```text
resolved actual_kid
  -> 是否需要 workspace
  -> workspace family / shape
  -> workspace BF16或FP32 dtype
  -> launch split-K或compile-time fused SplitK
  -> generated launcher
```

workspace 初始化代码不再查询 CSV、不再执行 heuristic，也不会重新选择 kid。

### 1.3 逐阶段直接对比

| 对比项 | 最初版本 `ca68b4f...` | 当前工作树 | 实际变化 |
|---|---|---|---|
| Python选择入口 | explicit；CSV命中；CSV miss转 C++ | explicit → tuned CSV → Python heuristic → framework fallback | 选择权统一到 Python |
| CSV miss | `_opus_gemm_bf16_dispatch` → C++ lookup/heuristic | Python architecture heuristic | 不再依赖 generic BF16 C++入口二次选择 |
| selector结果 | 不存在统一的 resolved对象 | `LaunchConfig`，含 `actual_kid` 和三种 split-K值 | 新增单一解析结果 |
| workspace决定时机 | 进入 generated launcher之后 | raw调用之前 | 从 C++内部前移到 Python调用边界 |
| raw tune ABI | `(XQ,WQ,Y,bias,kernelId,splitK)` | `(XQ,WQ,Y,bias,workspace,kernelId,splitK)` | 新增 optional Tensor参数 |
| workspace owner | OPUS进程级 C++ registry | 当前调用持有的 Torch Tensor | 所有权交给框架 |
| stream状态 | `unordered_map<hipStream_t, Owner*>` + mutex | OPUS无 per-stream registry | 删除隐藏全局状态 |
| allocation | launcher内部 `hipMalloc/hipFree` | `torch.empty`或 caller-provided Tensor | 使用 Torch allocator/lifetime |
| grow策略 | 按 stream复用，容量不足时扩容并释放旧地址 | 每次自动分配，或 caller显式复用 | 删除 OPUS grow逻辑 |
| kernel取地址 | host handle / device mirror间接取得 | `workspace.data_ptr()`直接取得 | 删除 handle间接层 |
| capability判断 | C++ kid数值范围和分支 | generated table membership + Python `SPLITK_KIDS` | exact-kid、按架构隔离 |
| shape来源 | 各 generated launcher内部算 bytes/stride | `_init_a16w16_workspace()`算 Tensor shape，C++复算容量防御 | Python可见且 C++仍最终校验 |
| dtype来源 | 分散在各架构 traits/launcher中，部分注释失真 | exact kid显式 `splitk_workspace_dtype` | metadata、writer、reader、validator统一 |
| caller复用 | 不支持传 Tensor | low-level API可传 workspace Tensor | 新增显式复用能力 |
| graph准备 | 每个 capture stream先 init并预热最大 shape | 无 OPUS专用 init/prewarm协议 | 生命周期交给 Torch；实机 graph仍待验证 |
| `opus_gemm_workspace_init()` | 真正创建当前 stream handle | deprecated Python no-op | 保留调用兼容，删除 C++ binding |
| non-workspace kid | raw调用后直接 launch | workspace必须为 `None`，再直接 launch | 增加显式合同校验 |
| gfx1250 fused | 不存在 | 1378个 exact kid | 新增 #4246 fused family |

### 1.4 原版 graph capture为什么需要 init和预热

最初版本的 `hipMalloc/hipFree` 不能在 HIP graph capture 中安全执行，因此调用方必须：

1. 在将要 capture 的同一个 stream 上、eager模式下调用 `opus_gemm_workspace_init()`；
2. 在该 stream 上先运行预计最大的 GEMM，把隐藏 buffer扩到足够容量；
3. 再开始 capture；capture期间不能发生 grow。

原版还在 `aiter/tuned_gemm.py` 中维护 capture-stream预热、shape签名集合和 `_opus_ws_warmed_sigs`
状态。当前已删除这些 OPUS专用逻辑。当前 workspace 是 Torch Tensor；自动分配走 Torch allocator，
也可由 caller在 capture前显式创建并复用。这里表示**设计上不再要求 OPUS init/prewarm协议**，不等于
已经完成实际 graph capture/replay验证。

### 1.5 `WorkspacePlan`只是中间步骤，不是最初版本

迁移过程中曾在 HEAD `2352c46c` 引入：

```text
_workspace.py
  -> WorkspacePlan / allocate_workspace / validate_workspace

_workspace_a16w16.py
  -> plan_a16w16_workspace
```

当前又将这两层撤销，最终只保留
`gemm_op_a16w16.py::_init_a16w16_workspace()`。因此：

- 相对当前 HEAD，这两个文件显示为删除；
- 相对最初基线 `ca68b4f...`，它们在两个端点都不存在；
- 不能把“删除 `WorkspacePlan`”写成最初流程到当前流程的文件删除；它是中间方案新增后撤销；
- 从最初端点到当前端点，没有任何原始代码文件被整文件删除。

## 2. 架构和 workspace dtype的端点对比

这里统计的是 **external partial workspace**。A/B 仍是 BF16；表中的 BF16/FP32指 workspace
storage，不是输入 dtype，也不是说架构是否支持 BF16。

| 架构 / family | 最初版本 | 当前版本 | 变化 |
|---|---:|---:|---|
| gfx950 FlatMM two-stage | 48 FP32 | 48 FP32 | 保持物理 FP32合同 |
| gfx950其他 a16w16 | 92个不使用 external workspace | 相同 | split-barrier/persistent/mono/atomic等不分配 external workspace |
| gfx942 external-workspace | 3 BF16 + 5 FP32 | 3 BF16 + 5 FP32 | 保持 exact-kid dtype |
| gfx942其他 a16w16 | 14个不使用 external workspace | 相同 | 无 external partial Tensor |
| gfx1250 two-stage plain | 28 FP32 | 28 BF16 | 恢复 #4246 writer/reader/metadata合同 |
| gfx1250 two-stage clusterlaunch | 468 FP32 | 468 BF16 | 同上 |
| gfx1250 fused | 0 | 780 BF16 + 598 FP32 | 新增1378个 |
| gfx1250合计 | 496：BF16 0 / FP32 496 | 1874：BF16 1276 / FP32 598 | 当前同时需要 BF16和FP32 |

原版 `opus_gemm.cu` 中“all splitk kids force FP32”的概括不能替代物理 writer/reader审计；gfx942
原本就有 3 个 BF16 workspace kid。gfx1250原版全部 FP32是当时实现状态，不是 gfx1250的架构
能力限制。当前 two-stage 496个恢复为 BF16，fused则按 exact kid同时覆盖 BF16和FP32。

当前 shape合同：

```text
gfx950 / gfx942 two-stage:
  [allocation_split_k, batch, padded_M, padded_N]

gfx1250 two-stage:
  [allocation_split_k, padded_M, padded_N]

gfx1250 fused:
  [num_tiles_m, num_tiles_n, fuse_split_k - 1, B_M, B_N]
```

two-stage表示 main kernel先把每个 split的 partial写入 workspace，然后另启 standalone reduce
kernel；fused表示前 `SplitK-1` 个 workgroup写 tile-major partial，最后一个 workgroup在同一个
clustered kernel内完成 reduce并写 Y，不再启动第二个 reduce kernel。

## 3. 从最初基线到当前版本的完整文件规模

### 3.1 主统计

统计范围为 `ca68b4f...` 到当前工作树的代码文件，排除 `docs/`：

```text
tracked endpoint diff:
  44 files changed
  5114 insertions, 2752 deletions
  36 modified, 8 added, 0 deleted

加上当前未跟踪的 fused pipeline:
  45 code files
  5752 insertions, 2752 deletions
  36 modified, 9 added, 0 deleted
```

当前未跟踪 fused pipeline为638行。主统计之所以不能只写“17个文件”，是因为17个文件只是
当前 HEAD `2352c46c` 之后的未提交增量，并不包含从最初 C++隐式 workspace版本到 HEAD 已完成
的 selector、raw ABI、direct pointer、validator和测试迁移。

次级统计（仅当前 HEAD到工作树）为：17个 tracked文件，`+1366/-731`，其中15个修改、2个
删除，另有1个638行的未跟踪 fused pipeline。这里的2个删除正是中间新增的 `_workspace.py` 和
`_workspace_a16w16.py`。

### 3.2 完整45个代码文件

| 状态 | 文件 | 基线→当前 | 端点变化职责 |
|---|---|---:|---|
| 修改 | `aiter/ops/opus/README.md` | `+239/-995` | 将旧 C++ registry/prewarm说明改为当前 selector和 Torch workspace接口说明 |
| 修改 | `aiter/ops/opus/__init__.py` | `+6/-1` | 保留导出；unsupported arch也转到 deprecated Python no-op |
| 新增 | `aiter/ops/opus/_selector_a16w16.py` | `+418/-0` | 统一 explicit、CSV、Python heuristic、fallback并解析最终 `actual_kid` |
| 修改 | `aiter/ops/opus/gemm_op_a16w16.py` | `+302/-172` | 增加 workspace raw参数、单一 init、Torch分配和统一 launch路径；删除 generic BF16生产分支 |
| 新增 | `aiter/ops/opus/heuristics/__init__.py` | `+40/-0` | 注册三架构 Python heuristic |
| 新增 | `aiter/ops/opus/heuristics/a16w16_gfx1250.py` | `+34/-0` | gfx1250 fallback heuristic |
| 新增 | `aiter/ops/opus/heuristics/a16w16_gfx942.py` | `+259/-0` | gfx942 Python heuristic迁入 |
| 新增 | `aiter/ops/opus/heuristics/a16w16_gfx950.py` | `+44/-0` | gfx950 Python heuristic迁入 |
| 修改 | `aiter/tuned_gemm.py` | `+0/-96` | 删除 per-stream workspace init、capture stream预热和 warmed-signature cache |
| 修改 | `csrc/include/rocm_ops.hpp` | `+4/-11` | pybind tune schema加入 workspace并删除 C++ workspace-init绑定宏 |
| 修改 | `csrc/opus_gemm/README.md` | `+154/-276` | 将隐藏 allocator/handle文档更新为 caller-owned direct-pointer合同 |
| 修改 | `csrc/opus_gemm/codegen/common.py` | `+22/-2` | 增加共享 workspace dtype映射和 fused tag支持 |
| 修改 | `csrc/opus_gemm/codegen/gen_instances_gfx1250.py` | `+312/-92` | 生成 typed two-stage/fused launcher、容量验证和 direct pointer |
| 修改 | `csrc/opus_gemm/codegen/gen_instances_gfx942.py` | `+57/-97` | 删除 registry grow模板，生成 typed caller-workspace launcher |
| 修改 | `csrc/opus_gemm/codegen/gen_instances_gfx950.py` | `+51/-56` | 删除 registry grow模板，生成 FP32 caller-workspace launcher |
| 修改 | `csrc/opus_gemm/gen_instances.py` | `+177/-120` | 生成5/6参数分表、workspace manifest、typed reducer和 fused TU |
| 修改 | `csrc/opus_gemm/include/gfx1250/opus_gemm_arch_gfx1250.cuh` | `+163/-99` | generated exact-kid non-workspace/workspace路由表 |
| 修改 | `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh` | `+11/-8` | plain two-stage按 `D_WS`写 BF16/FP32 direct workspace |
| 修改 | `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh` | `+11/-8` | clusterlaunch two-stage按 `D_WS`写 direct workspace |
| 修改 | `csrc/opus_gemm/include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh` | `+52/-31` | handle改 direct pointer；增加 BF16/FP32 `D_WS`和 fused kargs |
| 修改 | `csrc/opus_gemm/include/gfx1250/splitk_reduce_gfx1250.cuh` | `+44/-30` | BF16/FP32 partial读取后以FP32累加 |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_helpers_a16w16.cuh` | `+6/-6` | helper由 device handle读取改为 direct workspace pointer |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_em3en4_lds1_pgr2_sk.cuh` | `+3/-3` | split-K pipeline接收 direct pointer |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_kbuf1.cuh` | `+2/-1` | split-K store路径适配 direct pointer |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_kbuf2v.cuh` | `+2/-1` | split-K store路径适配 direct pointer |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_kbuf2v_bk128.cuh` | `+3/-1` | BK128 split-K路径适配 direct pointer |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_quad_mfma32_kbuf1.cuh` | `+10/-8` | quad split-K direct pointer和typed store |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_traits_a16w16.cuh` | `+18/-14` | workspace kargs从 handle改为直接地址 |
| 修改 | `csrc/opus_gemm/include/gfx942/a16w16/splitk_reduce_gfx942.cuh` | `+9/-9` | reducer从 direct typed workspace读取 |
| 修改 | `csrc/opus_gemm/include/gfx942/opus_gemm_arch_gfx942.cuh` | `+197/-99` | generated exact-kid路由和5/6参数 ABI |
| 修改 | `csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh` | `+138/-185` | generated membership表取代数值范围/隐藏 allocator路由 |
| 修改 | `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh` | `+2/-3` | main kernel直接使用 FP32 workspace pointer |
| 修改 | `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh` | `+7/-18` | 删除 workspace handle，kargs保存 direct pointer |
| 修改 | `csrc/opus_gemm/include/gfx950/splitk_reduce_gfx950.cuh` | `+5/-6` | reducer直接读取 caller-owned FP32 workspace |
| 修改 | `csrc/opus_gemm/include/opus_gemm.h` | `+1/-3` | tune声明加入 optional workspace；删除 C++ init声明 |
| 修改 | `csrc/opus_gemm/include/opus_gemm_common.cuh` | `+76/-2` | 新增 checked extent和 workspace物理合同 validator |
| 修改 | `csrc/opus_gemm/opus_gemm.cu` | `+76/-252` | 删除 registry/handle/init/C++ BF16 heuristic生产入口；按 generated membership分发 |
| 修改 | `csrc/opus_gemm/opus_gemm_common.py` | `+213/-10` | exact-kid capability/dtype metadata、gfx1250 fused registry和 invariant审计 |
| 修改 | `csrc/opus_gemm/opus_gemm_tune.py` | `+191/-27` | tuner接入 fused、per-dtype compile-time SplitK和容量/bias过滤 |
| 修改 | `csrc/pybind/opus_gemm_pybind.cu` | `+0/-1` | 删除 `OPUS_GEMM_WORKSPACE_INIT_PYBIND` 注册 |
| 修改 | `op_tests/test_opus_a16w16_gemm.py` | `+127/-9` | 更新公开/低级API和 dispatch行为测试 |
| 新增 | `op_tests/test_opus_dispatch.py` | `+607/-0` | 覆盖选择优先级、redirect、split-K和 fallback |
| 新增 | `op_tests/test_opus_graph.py` | `+192/-0` | 覆盖无 registry/cache的生命周期合同及 GPU graph用例 |
| 新增 | `op_tests/test_opus_workspace.py` | `+829/-0` | 覆盖 dtype、shape、ABI、validator、codegen和 fused registry |
| 新增 | `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_fuse_gfx1250.cuh` | `+638/-0` | #4246 single-kernel fused split-K pipeline |

## 4. 删除、替换和新增了什么

### 4.1 从最初设计中真正删除的机制

| 原机制 | 删除位置 | 当前替代 |
|---|---|---|
| `SplitkWsRegistry`和 process-global mutex map | `csrc/opus_gemm/opus_gemm.cu` | 调用级 Torch Tensor |
| `opus_splitk_ws_get()` | `opus_gemm.cu`及三架构 generated launcher | `_init_a16w16_workspace()` + raw参数 |
| host-pinned `opus_splitk_ws_handle` | gfx950/gfx1250 traits及 common glue | kernel kargs中的 direct pointer |
| gfx942 device handle mirror和同步 | `opus_gemm.cu`/gfx942 launcher | direct `workspace.data_ptr()` |
| launcher内 `hipMalloc/hipFree` grow | `gen_instances_gfx950/942/1250.py`模板 | `torch.empty`或 caller Tensor |
| C++ `opus_gemm_workspace_init()` | header、implementation、pybind宏/注册 | deprecated Python no-op |
| `tuned_gemm` capture stream预热 | `aiter/tuned_gemm.py` | 无 OPUS专用预热状态 |
| generic BF16 C++生产选择 | Python `_opus_gemm_bf16_dispatch`调用和 C++ BF16分支 | Python `select_launch_config()` |
| split-K kid数值区间作为 ABI真值 | `opus_gemm.cu` | generated per-arch membership表 |

没有删除 generic `opus_gemm` 整个函数，因为 a8w8仍使用它；只是它的 BF16 a16w16 generic
dispatch已禁用，a16w16必须走 Python已解析的 tune入口。

### 4.2 当前新增的核心能力

1. `select_launch_config()`把 explicit、tuned、heuristic和 fallback收敛为唯一 `LaunchConfig`；
2. 无效 tuned row会原子丢弃 kid和 split-K，再进入 heuristic，不混用旧字段；
3. requested kid可安全 redirect到最终 `actual_kid`，workspace只看最终值；
4. `_init_a16w16_workspace()`在一个现有 Python文件内完成 capability、shape、dtype和分配；
5. low-level caller可提供 workspace Tensor并原样复用；
6. C++ generated ABI明确区分 non-workspace五参数表和 workspace六参数表；
7. C++ validator检查 device、dtype、contiguous、16-byte alignment、capacity和 extent overflow；
8. 所有 external-workspace kid必须显式声明 `bf16_t`或`fp32_t`；
9. gfx1250 two-stage writer、reader和 metadata同步为 BF16；
10. gfx1250新增1378个 fused kid，并支持BF16/FP32 tile-major workspace；
11. fused tuner按 `N-cluster × workspace dtype`分别选择 compile-time SplitK；
12. framework fallback成为 selector的显式终点，不再隐式进入另一套 C++选择器。

### 4.3 单一 workspace init具体做什么

`_init_a16w16_workspace()`只接受已经解析完成的 `LaunchConfig`：

1. 用 `(arch, family, actual_kid)`定位 canonical instance；
2. 以 `actual_kid in SPLITK_KIDS`判断 external workspace capability；
3. non-workspace kid要求 caller workspace为 `None`；
4. 从 exact instance读取 `B_M/B_N/B_K`和 `splitk_workspace_dtype`；
5. 检查 runtime/effective SplitK为正且不超过 K tile上限；
6. two-stage计算 split-major shape；fused计算 tile-major shape；
7. 在 `torch.empty` 前检查正 extent、numel和字节乘法溢出；
8. caller传 Tensor时不在 Python重复做物理验证，交给 generated C++最终校验；
9. caller未传时执行一次 `torch.empty(shape,dtype,device)`；
10. 不使用全局 Tensor cache，也不跨调用保存隐式 workspace。

## 5. 当前 C++和 kernel合同

### 5.1 raw与 generated dispatch

最初 raw tune接口：

```text
_opus_gemm_a16w16_tune_raw(
  XQ, WQ, Y, bias, kernelId, splitK
)
```

当前 raw tune接口：

```text
_opus_gemm_a16w16_tune_raw(
  XQ, WQ, Y, bias, workspace, actual_kid, launch_split_k
)
```

`opus_gemm_a16w16_tune()`在 C++ 中先查询当前架构的 generated workspace membership：

- membership命中：workspace必须存在，调用 workspace launcher；
- membership未命中：workspace必须为 `None`，调用 non-workspace launcher；
- 不再根据跨架构共享的 kid数值范围猜测 ABI。

### 5.2 generated C++ validator

workspace launcher会复算 exact required capacity，并验证：

- workspace与 XQ在同一 device；
- dtype等于 exact kid要求的 BF16或FP32；
- contiguous；
- `numel >= required_numel`；
- data pointer非空且至少16-byte aligned；
- extent product和 byte size不溢出。

通过后只把 `workspace.data_ptr()`写入 kargs。main writer和 reducer reader都使用由同一 metadata
生成的 storage type。

### 5.3 gfx1250 two-stage与 fused

gfx1250 two-stage当前包括 plain 28个和 clusterlaunch 468个：

```text
main kernel: FP32 accumulator -> cast BF16 -> split-major workspace
standalone reduce: BF16 partial -> FP32 accumulation -> bias -> BF16/FP32 Y
```

gfx1250 fused当前包括 kid `21000..22377` 共1378个：

```text
cluster = (compile-time SplitK, N-direction peers, 1)
前 SplitK-1 个 WG -> BF16或FP32 tile-major workspace
cluster barrier
最后 WG -> 读取 partial并以FP32累加 -> 直接写 Y
无 standalone reduce launch
```

fused当前要求 batch=1、K为偶数、N完整填充 `B_N` tile和 N cluster，ragged M可用；当前
`bias=True`因 tuned schema不能精确表达 bias dtype/shape而被 selector/tuner安全排除。

## 6. 四层职责变化

| 层 | 最初职责 | 当前职责 |
|---|---|---|
| Python API | 显式/CSV选择；miss交给 C++；看不到 workspace | 完成所有选择；创建/接收 workspace Tensor |
| Metadata/codegen | 生成 launcher；类型和 allocator逻辑分散 | exact-kid capability/dtype真值；生成5/6参数表和 validator |
| C++ dispatcher | C++ heuristic、数值范围、stream registry、workspace owner | 只按 generated membership路由并验证 caller Tensor |
| Kernel | 通过 handle取得隐藏 buffer | 直接接收 typed workspace pointer |

这也是本次迁移实现的核心功能：不是简单把 `hipMalloc`换成 `torch.empty`，而是把**选择、
capability、shape、dtype、所有权、ABI、验证和 kernel地址传递**全部改为同一个 actual-kid-first
合同。

## 7. 中间方案与端点文件状态说明

为了避免再次混淆，三个时间点如下：

```text
最初 ca68b4f
  C++ registry + hipMalloc/hipFree
  无 _workspace.py / _workspace_a16w16.py

中间 HEAD 2352c46c
  已切 Torch workspace
  有 WorkspacePlan + 两个专用 Python模块

当前工作树
  仍是 Torch workspace
  撤销 WorkspacePlan和两个专用模块
  只保留 _init_a16w16_workspace()
  加入 gfx1250 two-stage BF16和 fused BF16/FP32
```

因此“当前删除两个 `_workspace*` 文件”只适用于 **HEAD→当前工作树**；“最初→当前”的端点
文件表中它们既不是新增，也不是删除。

## 8. 已完成验证

CPU选择集：

```text
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  -k 'not raw_cpp and not graph_capture and not two_streams'

149 passed, 18 deselected, 0 failed
```

18个 deselected是明确过滤且没有执行的 raw GPU、graph capture/replay和双-stream用例，不是失败。

gfx950实机选择集（两张空闲 MI355X，物理 GPU 4、5）：

```text
HIP_VISIBLE_DEVICES=4,5 pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py

162 passed, 14 skipped, 0 failed
```

14项因需要 gfx942/gfx1250而跳过。gfx950实际通过了 split-K BF16/FP32输出和bias数值、raw
workspace正反合同、跨device拒绝、无prewarm graph capture/replay以及双stream不同调用级
workspace。测试结束后 GPU 4、5恢复0%利用率和0%显存占用，无测试残留KFD进程。

gfx950 canonical 全量选择集随后在隔离 JIT中构建，并用 physical GPU 4--7各跑一个 shard：

```text
140 canonical a16w16 kids
  48 external-workspace kids
  92 non-workspace kids

aggregate: 130 passed, 10 failed
```

48/48 workspace kid（`200--223`、`1200--1223`）全部通过 BF16/FP32 Y、caller复用和
Torch自动分配/生命周期；这证明当前 Torch-owned split-K workspace路径在 gfx950上已覆盖
全部登记 kid。10个失败全部是 non-workspace mono-tile `1400--1404`、`6400--6404`：BF16 Y
先通过，FP32 Y随后数值不匹配。最初基线复现因旧 Docker容器被终止而未完成，因此这10项尚未
归类为既有缺陷或当前 non-workspace回归，不能写成全140 kid零失败。

其他已完成验证：

- 相关 Python `py_compile`通过；
- canonical registry数量和 dtype审计通过；
- 全量 fused codegen生成1378个 impl header、2756个 device TU、1378个 lookup row；
- fused-only生成0个 standalone reduce TU；
- fresh组合 subset的 host、`opus_gemm.cu`、pybind和14个代表/边界 device/reduce TU通过 gfx1250
  `hipcc -fsyntax-only`；
- `git diff --check`通过。

## 9. 尚未完成的实机边界

- 尚未运行 gfx1250 fused GPU数值测试；
- gfx950实际 graph capture/replay和双stream已通过；gfx942/gfx1250对应路径尚未实机运行；
- 尚未运行 gfx1250 fused graph capture/replay和双stream；
- 尚未运行 fused性能调优/benchmark；
- gfx950 mono-tile `1400--1404`、`6400--6404` 的 FP32失败尚未完成 `ca68b4f...` 基线
  复现和归因；
- 尚未完成 gfx950原始/当前性能 A/B；
- 全量2756个 fused device TU已生成，但未逐一 HIP syntax编译；
- gfx1250 workspace路径继续要求 batch=1；
- fused `bias=True`当前有意关闭；
- 当前代码修改尚未提交；
- 本轮文档更新没有访问、终止或干扰未知 GPU/KFD进程。

因此当前可以确认的是：原始 C++隐式 workspace机制已经替换为 actual-kid驱动的 Torch Tensor
流程，CPU回归、metadata/codegen审计、代表性语法验证、gfx950 focused实机数值/合同/graph/
双stream，以及 gfx950全部48个 workspace kid均通过；不能把 gfx950 workspace结果扩写成
全140个 gfx950 kid零回归，也不能外推为 gfx942/gfx1250，更不能写成 gfx1250 fused数值、
graph、并发或性能已经实机通过。

## 10. 统计复现命令

tracked endpoint diff：

```bash
git diff --name-status ca68b4f3501762c15c550cb920a5516e9710cf89 -- ':!docs'
git diff --numstat ca68b4f3501762c15c550cb920a5516e9710cf89 -- ':!docs'
git diff --shortstat ca68b4f3501762c15c550cb920a5516e9710cf89 -- ':!docs'
```

上述命令不会显示未跟踪的638行 fused pipeline，完整45文件统计需要将该文件单独计入。
