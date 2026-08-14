# OPUS GEMM 当前架构简报

更新时间：`2026-08-13`

核心原则：**调用方给出最终 kid；Python 直接查询基线已有的 `kernels_list`，再由现有
family模块校验dtype/layout并管理Torch workspace；C++只做family exact-kid dispatch和
kernel launch。**

## 总体数据流

```text
调用方 / tuner / 高层AITER dispatcher
  │  A16: explicit / tuned CSV / per-arch heuristic / PyTorch fallback
  │  进入OPUS时已解析最终 kid，已分配 Y
  ▼
aiter.ops.opus.opus_gemm(..., kid=...)
  │
  ├─ 基线已有的 kernels_list.get(kid) -> instance
  ├─ instance arch/tag决定进入A16或A8现有模块
  ├─ family模块检查XQ/WQ/Y dtype与plain/bpreshuffle声明
  ├─ family模块检查scale/bias/workspace/split_k参数
  └─ 调用私有family adapter
       │
       ├─ a16w16: exact-kid legality + scalar plan cache + Torch workspace
       ├─ A8 GEMM families: exact capability check
       └─ A8 MXFP8 BMM: exact capability + Torch workspace
               │
               ▼
C++ family raw entry
  -> 当前arch/output dtype typed table
  -> exact kid lookup
  -> generated launcher checks
  -> kernel [-> reduce]
```

以下内容不在exact public与C++ runtime路径：

- shape-driven selector；
- tuned CSV lookup；
- Python/C++ heuristic；
- default kid；
- requested kid到actual kid重定向；
- framework fallback；
- 按kid数字区间猜family。

A16上层caller仍运行tuned CSV和现有A16文件内的private per-arch heuristic；它们在public调用
之前结束，只把最终kid传入上述exact路径。C++ runtime heuristic不恢复。

## Python公开面

```python
opus_gemm(
    XQ,
    WQ,
    Y,
    *,
    kid,
    layout="plain",
    x_scale=None,
    w_scale=None,
    bias=None,
    split_k=0,
    workspace=None,
)
```

公开 `__all__` 只有 `opus_gemm`。四个A8 family adapter和A16 adapter均为私有实现，
family模块的 `__all__` 为空。`Y`由调用方持有并原样返回。

## Family合同

| Family | 当前arch/kid | Python侧关键输入规则 |
|---|---|---|
| A16W16 | gfx942/gfx950/gfx1250 | BF16输入，BF16/FP32输出，plain layout，可选bias/split-K/workspace |
| A8W8 | gfx950 kid 2 | FP8输入、FP32输出、无scale |
| A8W8 blockscale | gfx950 kid 1 | FP8输入、FP32输出、plain WQ、两份FP32 scale |
| A8W8 blockscale bpreshuffle | gfx942 kid 11000 | FP8输入、BF16输出、bpreshuffle WQ、两份FP32 scale |
| A8W8 MXFP8 BMM | gfx950 45个global kid（8000--8653） | `[M,G,K] × [G,N,K]`、E8M0 scale、BF16/FP32输出、可选split-K workspace |

不同arch的kid编号段和合并后的`kernels_list`在原始基线中已经存在，并非Task2新增。
PR #4320的45个BMM id按`8000 + upstream kid`加入同一dict；当前总数为2084。Task2的
唯一public入口直接复用该dict，不新增
per-arch Python路由文件，也不重新编号。

## Task1 workspace

### 改造前

```text
C++ launcher
  -> per-stream registry
  -> hipMalloc/hipFree grow
  -> host/device handle mirror
  -> graph prewarm
  -> device二次解引用workspace指针
```

### 当前

```text
exact kid + split_k
  -> immutable scalar workspace plan
  -> caller workspace或每次torch.empty
  -> C ABI传递Tensor
  -> generated launcher校验
  -> direct ptr_ws
```

public contract cache最多4096项，只保存route/dtype/layout/option presence；A16 launch-plan cache
最多256项，只保存kid、split、shape和dtype等不可变标量。两者都不保存Tensor、地址、
workspace、stream、allocation或launcher状态。每次调用仍检查live layout并进入C++ checked
validator。

| 架构 | Workspace布局 |
|---|---|
| gfx950 two-stage | `[S,batch,padded_M,padded_N]`, FP32 |
| gfx942 two-stage | `[S,batch,padded_M,padded_N]`, exact BF16/FP32 |
| gfx1250 two-stage | `[S,padded_M,padded_N]`, BF16 |
| gfx1250 fused | `[tiles_m,tiles_n,fuse_split_k-1,B_M,B_N]`, exact BF16/FP32 |

显式传给public的gfx942 BF16-workspace kid 10210/10213/10216在非exact-N时直接拒绝。tuned或
heuristic候选的legacy requested-to-actual解析发生在上层caller，解析后的最终kid才进入public。

## MXFP8 BMM workspace

PR #4320的BMM family复用相同所有权模型：`gemm_op_a8w8.py`按exact kid和shape计算FP32
容量，复用caller Tensor或创建本次调用的`torch.empty`，然后把直接指针交给
`opus_gemm_a8w8_mxscale_bmm_launch`。two-stage布局保存
`split_k × G × padded_M × padded_N`个partial；fused布局在同一个FP32 Tensor尾部放置
对齐后的tile counter。C++和GPU reducer都不持有Tensor，也不分配workspace。

## C++边界

C++保留五个family entry和A16 status-returning C ABI。它们不拥有policy、Tensor或workspace。
pybind raw负责既有lazy JIT build并作为私有性能A-B端点保留；mixed-module ctypes适配局限在
已有A16 Python文件内，`aiter/jit/core.py`与Task1原始基线零diff。generated launcher继续
负责最终shape、stride、dtype、device、alignment、capacity、bias和tile安全检查。

## Build-time和runtime分离

CSV与compiled-kids sidecar决定subset `.so`编译哪些exact id。`DEFAULT_COMPILED_KIDS*`既是
compile floor，也保证A16 caller-side heuristic的全部结果可用；public/C++不使用它做shape
选择。mandatory A8 ids仍为gfx950 `{1,2}`和gfx942 `{11000}`。gfx950 BMM的45个exact route
作为一个family全部生成，按symbol去重，不依赖普通per-kid subset集合。

## 当前文件结构

```text
aiter/ops/opus/
├─ __init__.py              # 唯一public router
├─ _arch.py                 # 显式device维度的arch/CU scalar cache
├─ gemm_op_a16w16.py        # private A16 heuristic、exact launch、Torch workspace、局部C ABI
└─ gemm_op_a8w8.py          # 四个私有exact family adapter，含MXFP8 BMM workspace

aiter/ops/batched_gemm_op_a8w8.py
└─ tuned MXFP8 BMM调用方，最终仍进入统一opus_gemm

csrc/opus_gemm/
├─ opus_gemm_common.py      # 基线registry及Task1 workspace metadata
├─ opus_gemm.cu             # family router和A16 C ABI
├─ opus_bmm.cu              # MXFP8 BMM exact-kid family router
├─ include/opus_bmm.h       # BMM C++声明
├─ gen_instances.py         # exact-id subset/codegen
├─ generated kid tables
└─ gfx950 BMM traits / pipeline / reduce
```

## 性能结论

接口收敛前的Task1首尾数据仍记录在性能计划第10节；统一入口和局部C ABI的首尾/增量数据见
第11、12节。最终改成直接复用基线`kernels_list`后又完成相邻防回退（第13节）：public eager
`1496.886513 -> 1497.065125 us`，变化`+0.011932%`且`48快/48慢`；graph变化
`+0.094053%`且不执行Python查询。两项均判定为测量噪声范围内持平。

PR #4320接入后的同机接口复测（物理GPU 4，MI355X，20 warmup、9×100、两轮）表明：
A16 direct和graph相对接入前Task2模块分别变化`-0.216%`和`-0.177%`；统一public eager的
额外route成本仍约5%。BMM在18个DSV4代表shape上，tuned unified路径相对固定global kid 8000
的累计耗时下降`60.51%`（token-major）和`60.93%`（batch-major）。
