# OPUS GEMM Task2 当前文件与功能

更新时间：`2026-08-13 UTC`

本文只描述当前源码。Task2 的结果是一个 public `opus_gemm`，按调用方传入的最终
`kid` 查询统一 `kernels_list`，再进入 private family launcher。PR #4320 的
gfx950 MXFP8 BMM 已按同一结构接入现有 A8W8 路径。

## 1. 当前公开接口

```python
from aiter.ops.opus import opus_gemm

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

- `Y` 和最终 `kid` 必传；
- `aiter.ops.opus.__all__ == ["opus_gemm"]`；
- exact public/runtime 不选择另一个 kid，不 fallback，不 redirect；
- A16 上层调用方保持 `explicit -> tuned CSV -> per-arch heuristic -> PyTorch fallback`；
- public router 直接使用 `kernels_list.get(kid)`；
- `aiter/jit/core.py` 不包含 OPUS 特例，当前内容 hash 为
  `43231ab3cd9ea24caaa6e8535b71455386dbe0f5`。

## 2. 当前 family

| Logical family | Private Python launcher | C++ family entry | 当前能力 |
|---|---|---|---|
| `a16w16` | `_launch_a16w16` | `opus_gemm_a16w16_launch` | gfx942/gfx950/gfx1250，Torch split-K workspace |
| `a8w8` | `_launch_a8w8` | `opus_gemm_a8w8_launch` | gfx950 kid 2 |
| `a8w8_blockscale` | `_launch_a8w8_blockscale` | `opus_gemm_a8w8_blockscale_launch` | gfx950 kid 1 |
| `a8w8_blockscale_bpreshuffle` | `_launch_a8w8_blockscale_bpreshuffle` | `opus_gemm_a8w8_blockscale_bpreshuffle_launch` | gfx942 kid 11000 |
| `a8w8_mxscale_bmm` | `_launch_a8w8_mxscale_bmm` | `opus_gemm_a8w8_mxscale_bmm_launch` | gfx950 45 个 global kid，BF16/FP32 Y |

MXFP8 BMM 接受 `layout="mxscale_bmm"`，并兼容 `mxfp8_bmm` 和
`bmm_mxscale` 两个 alias。上游 PR #4320 的 id 映射为：

```text
global_kid = 8000 + upstream_kid
45 个注册 id，最小 8000，最大 8653
```

统一 registry 当前为 2084 项。BMM route 与已有 kid 不冲突，且仍由
`get_kernel_instance("gfx950", "a8w8_mxscale_bmm", kid)` 做 exact 查询。

## 3. Python 文件

| 当前文件 | 职责 |
|---|---|
| `aiter/ops/opus/__init__.py` | 唯一 public router、参数规范化、直接 registry 查询、五个 family 分派 |
| `aiter/ops/opus/_arch.py` | 按显式 device 获取 arch/CU 标量信息 |
| `aiter/ops/opus/gemm_op_a16w16.py` | private 三架构 A16 heuristic、caller candidate 最终 kid 解析、exact-kid、Torch workspace、private raw/C ABI |
| `aiter/tuned_gemm.py` | A16 tuned row 校验；无有效 row 时解析 per-arch heuristic，得到最终 kid 后调用统一 public，否则 PyTorch fallback |
| `aiter/ops/opus/gemm_op_a8w8.py` | 四个 A8 private launcher；BMM shape/stride、plan、workspace 创建/复用 |
| `aiter/ops/batched_gemm_op_a8w8.py` | MXFP8 tuned CSV、padded-M lookup、最终 global kid、统一 public 调用 |
| `aiter/configs/model_configs/dsv4_batched_gemm_a8w8_blockscale_mxscale_tuned.csv` | DSV4 gfx950 global-kid tuned rows |
| `aiter/jit/optCompilerConfig.json` | `module_deepgemm_opus` 同时编译 `opus_gemm.cu` 与 `opus_bmm.cu` |

没有新增生产 Python interface 文件。BMM raw binding 和 launcher 均在已有
`gemm_op_a8w8.py`；高层调用方在已有 `batched_gemm_op_a8w8.py`。

## 4. C++、codegen 与 GPU 文件

| 当前文件 | 职责 |
|---|---|
| `csrc/include/rocm_ops.hpp` / `csrc/pybind/opus_gemm_pybind.cu` | 五个 private family pybind schema/registration |
| `csrc/opus_gemm/include/opus_bmm.h` / `csrc/opus_gemm/opus_bmm.cu` | BMM exact-kid C++ entry、gfx950 gate、direct workspace forwarding |
| `csrc/opus_gemm/opus_gemm_common.py` | 45 个 BMM instance、global id 映射、统一 registry metadata |
| `csrc/opus_gemm/gen_instances.py` | BMM manifest、45-entry dispatch、全 family codegen |
| `csrc/opus_gemm/codegen/gen_instances_gfx950.py` | 八类 BMM instance launcher、M-alignment invariant、reduce TU |
| `csrc/opus_gemm/include/gfx950/opus_bmm_launchers_a8w8_mxscale_gfx950.cuh` | BMM shared host declarations/checks |
| `csrc/opus_gemm/include/gfx950/opus_bmm_pipeline_a8w8_mxscale_gfx950.cuh` | specialized BMM pipelines |
| `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh` | flatmm/fused split-K main pipeline |
| `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_scale_gfx950.cuh` | shared A8 device layout helpers |
| `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh` | A8/MXFP8 traits and direct workspace kargs |
| `csrc/opus_gemm/include/gfx950/splitk_reduce_gfx950.cuh` | A16/BMM direct-pointer reduce kernels |
| `csrc/opus_gemm/include/opus_gemm_utils.cuh` | BMM所需共享类型与工具 |
| `csrc/opus_gemm/opus_bmm_mxscale_tune.py` | 45-kid tuner，计时前准备 Torch workspace |

generated BMM 表为 `opus_bmm_mxscale_kid_dispatch.h`，大小固定为 45。unknown
kid 立即报错；C++ 不选择 baseline kid。

## 5. MXFP8 BMM Torch workspace

`gemm_op_a8w8.py` 的 scalar-only plan 根据 exact instance、M/G/N/K 和 split-K
返回有效 split 与所需 FP32 元素数：

- two-stage：`split_k * G * padded_M * padded_N`；
- fused：partial 区域后按 256-byte 对齐放置 tile counter；
- `split_k == 1`：不创建 workspace；
- direct-only kid 8646：只接受 split 1；
- caller Tensor：必须同 device、FP32、contiguous 且容量足够。

Python 自动创建的 Tensor 只由当前调用持有；caller Tensor 可显式复用。C++ launcher、
GPU main kernel 和 reduce kernel只接收 direct pointer，不保存 Tensor 或地址。

## 6. 调用链

```text
caller / tuned A8W8 dispatcher
  -> final global kid + caller-owned Y
  -> aiter.ops.opus.opus_gemm
  -> kernels_list.get(kid)
  -> _launch_a8w8_mxscale_bmm
  -> scalar plan + caller/automatic Torch workspace
  -> private pybind raw
  -> opus_gemm_a8w8_mxscale_bmm_launch
  -> generated 45-entry exact table
  -> selected BMM kernel [-> reduce]
```

## 7. 测试与性能

| 覆盖 | 当前结果 |
|---|---|
| focused OPUS suite | `126 passed, 17 skipped` |
| gfx950 A16 canonical sweep | `140 passed` |
| MXFP8 BMM M-alignment/exact launcher | 45/45 kid |
| BMM tile-N / token-major / batch-major | 通过 |
| BMM automatic workspace / caller workspace / Graph replay | 通过 |
| config collision tests | `13 passed` |
| fresh JIT | `/tmp/aiter-pr4320-bmm-fresh2.r7Buc2` |

在 MI355X 物理 GPU 4 上，18 个代表 shape
`G={2,8,16}, M={1,16,128,512,2048,8192}, N=1024, K=4096`：

| 路径 | 固定 kid 8000 累计 | tuned/统一路径累计 | 变化 |
|---|---:|---:|---:|
| token-major high-level | 3640.135 µs | 1437.482 µs | -60.51% |
| batch-major view unified public | 3608.364 µs | 1409.615 µs | -60.93% |

接口与 BMM 性能日志位于 `/tmp/aiter-pr4320-final-20260813`。

## 8. 最终边界

- public Python API 只有 `opus_gemm`；
- family launcher 均为 private；
- final kid 由 caller 提供；A16 caller 按 explicit/tuned/heuristic/fallback 优先级解析；
- exact public/runtime 直接查统一 `kernels_list`；
- A16 和 BMM split-K workspace 均由 Torch Tensor 持有；
- `aiter/jit/core.py` 内容保持原始 Task1 基线。
