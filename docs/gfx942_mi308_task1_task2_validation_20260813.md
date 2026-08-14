# OPUS Task1 / Task2 gfx942 MI308 实机验证结果

执行时间：`2026-08-13T16:34:16Z` 至 `2026-08-13T16:55:17Z`

执行方案：`docs/gfx942_mi300_mi308_task1_task2_validation.md`

结果目录：`/tmp/aiter-gfx942-mi308-e426.iUoZaY`

## 1. 结论

MI308 的 gfx942 合法硬件合同已完成正确性、Graph、双 stream、跨 device 和
Task1/Task2 ABBA 性能验证。执行中发现并修复两个真实 kernel 缺陷：

1. kid 10000 的 FP32 full-tile 输出使用了容量不足的 BF16 A/B LDS 暂存区；
2. kid 10310–10314 在 kernel 内清零输出，与其他 split block 的 atomic add 存在竞态。

修复后 current focused 为 `145 passed`，A16 canonical kid 为 `22/22`，A8 kid
11000 的 raw、unified public、真实 tuned CSV、Graph 和双 stream 全部通过。

原方案的“8 个 workspace kid × BF16/FP32 Y = 16 case”与 gfx942 launcher 的物理
合同冲突。kid 10210、10213、10216 使用 BF16 workspace，并明确只支持 BF16 Y；
原始 baseline 和 current 均以相同错误拒绝其 FP32 Y。因此可执行集合为 `13/13`
数值 case，另有 `3/3` FP32 拒绝 case。若严格按原文要求 `16/16`，MI308 不能签为
字面意义上的完整通过；必须先把方案修正为 13 个合法数值 case 加 3 个拒绝 case。

## 2. 环境与冻结端点

- GPU：AMD Instinct MI308X
- runtime arch：`gfx942:sramecc+:xnack-`
- CU：80
- 正确性物理 GPU：0；性能物理 GPU：5；进程内均为 GPU 0
- Python：3.10.12
- PyTorch：2.9.1+rocm7.2.0.git7e1940d4
- HIP runtime：7.2.26015-fc0010cf6a
- hipcc：HIP 7.2.26015-fc0010cf6a，clang 22.0.0git
- CPU core：0
- current HEAD：`e42611aeb0d938bdae8799e34d911fd3086dea41`
- baseline：`ca68b4f3501762c15c550cb920a5516e9710cf89`
- Task1 frozen：`2352c46c784d6ba3a0c71ff89b4bdb4c2fefa59f`
- sidecar SHA-256：`b1b8e2b7c18834be20cdd0a9425c18bb5d22ef4a50026fbca2258aea62e2bfae`
- baseline module SHA-256：`98094b1fd8a3053cdde12f7531e896d5f37e44e616f7dd729eb5f752c499986d`
- Task1 module SHA-256：`4d4299c0476079b8718fb628642e4ba6c6109beba100345af7cdc4d386698ceb`
- fixed current module SHA-256：`9801d6a251d0969d029a7a5985725fa8224d8d57c9cafac0472744583798b833`
- final tracked patch SHA-256：`303a55079061c8143a2112efec3154dae2afef0555da35d69192c47829f28062`

三个 module 均为目标节点 fresh build；offload audit 只包含 gfx942。

## 3. 正确性结果

- current focused：`145 passed, 33 skipped, 0 failed`
  - 33 个 skip 均为 gfx950、gfx1250 或多卡专属项；
  - 没有 gfx942 hardware skip。
- current exhaustive：`25 passed, 0 skipped, 0 failed`
  - 22 个 A16 kid 全部通过；
  - workspace kid `8/8`，合法数值 case `13/13`；
  - non-workspace kid `14/14`；
  - BF16-workspace FP32 Y 拒绝 `3/3`。
- baseline workspace：合法数值 case `13/13`，FP32 Y 拒绝 `3/3`。
- A16 kid 10200/10210：caller workspace、Graph、双 stream 全部通过。
- A8 kid 11000：2D raw、batch=1 3D raw、unified public、真实临时 tuned CSV、
  Graph、双 stream 和错误合同负例全部通过。
- 跨 device workspace：`1 passed`。
- Python tuned、invalid tuned、heuristic、fallback、redirect policy 均包含在 focused
  suite 并通过。

## 4. Task1 ABBA 性能

口径：GPU5，CPU core 0，warmup 20，9 rounds，每 round 100 次；
顺序固定为 baseline A1、current B1、current B2、baseline A2。以下均为 13 个合法
workspace case 的逐 case median 求和，再对 A1/A2、B1/B2 配对平均。

- raw/C ABI：
  - baseline eager：401.598 us；current：342.123 us，`-14.810%`
  - baseline Graph：388.097 us；current：385.097 us，`-0.773%`
  - 最大逐 case 退化：kid 10216 BF16 eager，`+0.259%`
  - 26 个 eager/Graph 比较中：24 快、2 慢
- 当前私有 A16 启动：
  - baseline eager：396.047 us；current：372.441 us，`-5.960%`
  - baseline Graph：386.646 us；current：384.956 us，`-0.437%`
  - 最大逐 case 退化：kid 10216 BF16 eager，`+0.634%`
  - 26 个 eager/Graph 比较中：24 快、2 慢

Task1 未发现超过本机重复漂移的 device/Graph 退化。

## 5. Task2 ABBA 性能

### 5.1 private 与 unified public

同一 fixed-current module、相同 caller workspace：

- private eager：369.543 us
- unified public eager：391.281 us，`+5.882%`
- private Graph：385.258 us
- unified public Graph：384.873 us，`-0.100%`
- 最大 eager route 成本：kid 10200 BF16，`+9.618%`

Graph 持平说明 kernel/device 工作未变化；eager 差异来自 Python unified route。

### 5.2 Task1 frozen 与 current 分层

- A16 kid 10200：
  - direct `+0.009%`，Graph `-0.119%`
  - raw `-32.311%`，private `-58.535%`，high-level `-55.743%`
- A16 kid 10210：
  - direct `+0.064%`，Graph `+0.098%`
  - raw `-32.409%`，private `-58.721%`，high-level `-55.583%`
- A8 kid 11000：
  - direct `+0.063%`，Graph `+0.085%`
  - raw `+3.227%`，private `+11.531%`，high-level `+25.854%`

A8 direct 和 Graph 在重复漂移内持平，说明 kid 11000 kernel 没有退化。A8 raw/private/
high-level 的增加是 host interface 与 unified Python route 成本，不能解释为 GPU kernel
退化；相关逐 round 数据保存在 Task2 ABBA 日志中。

## 6. 本次修改文件

生产代码：

- `csrc/opus_gemm/codegen/gen_instances_gfx942.py`
- `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_kbuf1_large_tile.cuh`
- `csrc/opus_gemm/include/gfx942/a16w16/opus_gemm_pipeline_a16w16_wave_k_coop.cuh`

测试与 benchmark：

- `op_tests/test_opus_gfx942_exhaustive.py`
- `op_tests/test_opus_gfx942_gpu.py`
- `op_tests/bench_opus_gfx950_workspace_ab.py`
- `op_tests/bench_opus_task1_task2_interfaces.py`
- `op_tests/test_opus_ctypes.py`

`test_opus_ctypes.py` 仅把显式 FakeTensorMode 下的 compile-visibility 测试改为
`backend="eager"`，避免 PyTorch 2.9 Inductor 创建第二个 FakeTensorMode；不降低
Dynamo fullgraph 检查。

## 7. 日志

完整日志、失败证据、fresh build、offload、逐 case PERF_CASE、ABBA 汇总和 module
位于：

`/tmp/aiter-gfx942-mi308-e426.iUoZaY`

失败日志未覆盖。`jit-current` 保存修复前 module，`jit-current-r1` 保存修复后 module。
