# GDN dense / packed-varlen prefill 适配说明

本适配层将 dense GDN prefill 拆为两个可独立构建的后端：

- W/U 后端：`opus_gdn_wu_prefill_fwd`，包含 WF（W/U fused）与 WS（W/U split）。
- C 后端：`opus_gdn_c_prefill_fwd`，包含 CF（C fused）与 CS（C split）。

统一入口为：

```python
from aiter.ops.gdn_prefill import gdn_prefill

o, final_state = gdn_prefill(
    q,
    k,
    v,
    o=output_buffer,
    g=g,
    beta=beta,
    initial_state=initial_state,
    output_final_state=True,
    path="auto",
)
```

接口参数与 `chunk_gated_delta_rule_opt_vk` 保持一致，只在末尾增加关键字参数
`path`。可选值为 `auto`、`cf`、`cs`、`wf`、`ws`、`c`、`wu` 和
`triton`。模型侧若已有自己的 shape 策略，可直接指定四条底层路径；`triton`
用于强制走原实现。

## Packed varlen

W/U split（WS）现已支持不补齐的 packed 输入。公开布局与
`chunk_gated_delta_rule_opt_vk` 一致：

- `q/k/v`: `[1, total_tokens, H, 128]` BF16；
- `g/beta`: `[1, total_tokens, H]` BF16 或 FP32；
- `cu_seqlens`: `[N+1]` int32/int64，严格递增，首尾为 0 和
  `total_tokens`；
- `initial_state/final_state`: `[N, H, 128, 128]` FP32（V/K 布局）；
- `o`: `[1, total_tokens, H, 128]` BF16。

每条序列及其最后一个非 64 对齐 chunk 都直接在 native kernel 中处理，不会
把序列 padding 后再调用 dense kernel。K1 和 K6 使用紧凑的全局 chunk 网格，
由 `(cu_seqlens, chunk_indices, chunk_offsets)` 映射到 sequence-local chunk；
K5 则由每个 workgroup 串行扫描一条序列。因此 gate cumsum、hidden state 和
chunk snapshot 均不会跨序列泄漏。由 `cu_seqlens` 派生的 int32 metadata 按
Tensor identity、mutation version 和 chunk size 缓存；同一个 Tensor 原地更新
后会重新校验和生成，不会复用旧的 snapshot offset。PyTorch inference tensor
没有 mutation version，因此 serving 中复用的 inference `cu_seqlens` 会按 identity
缓存，并视为生命周期内不可变；需要原地改写 offsets 的调用方必须使用普通、
带 version counter 的 Tensor。

HIP Graph capture 必须先在 capture 外用同一个 `cu_seqlens` 预热 metadata cache，
并在 graph 的整个生命周期保留这个静态 offsets Tensor。多项 weak-key cache 会
让内部 metadata 与对应 `cu_seqlens` 保持相同生命周期；因此一个 graph 可以预热
并使用多个 packed partition，GDN output 只作为 graph 中间值时也不受影响。
所有 replay 期间 offsets 都必须保持不变（普通 versioned Tensor 也一样）；packed
partition 发生变化时必须重新 capture，不能只在两次 replay 之间原地改 offsets。

性能路径会直接读取 BF16/FP32 gate，避免额外的 dtype-conversion kernel。所有
sequence 都 64 对齐时，K6 保留 varlen metadata 寻址但移除 token-tail predicate；
带 tail 时仍使用通用 K6。K5 还会利用 metadata 中的最长 sequence chunk 数，避免
低 head、高度偏斜的 ragged batch 让最长 sequence 的 state scan 欠占用。

`path="auto"` 在已验证的 gfx942/80-CU 环境选择 WS；`path="wu"` 和
`path="ws"` 可显式选择该路径（gfx942/gfx950）。packed varlen 暂不支持
CF/CS/WF，显式请求会报错；auto 遇到 GQA/MQA、BF16 state、decode prefix、
kernel 内 Q/K L2Norm、`use_exp2=False` 或其他未支持条件时仍原样回退 Triton。
WS 可复用与 `v` 完全相同的 output view；任何 partial overlap，以及 output 与
`q/k` 等后续仍需读取的输入共享 storage，都会被拒绝。

也可以直接调用 W/U 后端：

```python
o, final_state = opus_gdn_wu_prefill_fwd(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens=cu_seqlens,
    initial_state=initial_state,
    output_final_state=True,
)
```

## 性能基线

下面的数据在 AMD Instinct MI308X（gfx942、80 CU）、ROCm 7.14 上测得。
输入 gate 使用 mixed dtype（`g` 为 FP32、`beta` 为 BF16），packed metadata
使用普通、带 version counter 的 `cu_seqlens`，并在 cache hot 后计时。GPU event
样本按 provider 轮转及反向轮转，正确性检查、warmup 和 JIT 编译均不计入结果。
复现命令为：

```bash
python op_tests/triton_tests/bench_opus_gdn_prefill.py \
  --suite varlen --providers native-ws triton \
  --warmup 10 --repeat 40 --wall
```

Packed varlen 的 GPU event median 如下；Native 固定使用生产 WS 路径，Triton
为 AITER 的 `chunk_gated_delta_rule_opt_vk`，两边使用相同输入和各自独立的
预分配 output buffer：

短 ragged case 对共享机器上的频率状态更敏感；表中采用代表性 balanced run，
回归判断应同时查看 benchmark 输出的 p20/p80，而不是只比较单次 median。

| Packed lengths / heads | Native WS | AITER Triton | Native speedup |
|---|---:|---:|---:|
| `[63,64,65,511,1025]`, H16 | 215.34 us | 308.52 us | 1.43x |
| `[15,85,200,900]`, H4 | 118.34 us | 212.62 us | 1.80x |
| `[1024]x8`, H16（全对齐） | 748.14 us | 1328.95 us | 1.78x |
| 16 条边界长度、5969 tokens、H8 | 344.78 us | 563.14 us | 1.63x |
| `[1]x15+[8177]`, H4（高度偏斜） | 477.18 us | 832.63 us | 1.75x |
| 7809 tokens、H32、state I/O | 1658.07 us | 2687.83 us | 1.62x |

同一工具的 dense suite 会比较生产 `auto` 路由；代表点如下：

```bash
python op_tests/triton_tests/bench_opus_gdn_prefill.py \
  --suite dense --providers auto triton \
  --warmup 10 --repeat 40 --wall
```

| Dense shape | Auto route | Native auto | AITER Triton | Native speedup |
|---|---|---:|---:|---:|
| B1/T128/H8 | CS | 112.56 us | 257.06 us | 2.28x |
| B4/T2048/H16 | WF | 546.26 us | 1321.87 us | 2.42x |
| B1/T8192/H32/state I/O | WS | 1556.81 us | 2790.65 us | 1.79x |

`--wall` 还会输出每次调用前后同步的 caller latency，用于观察 metadata cache。
这些数字是特定软件栈与硬件上的基线，而不是跨平台承诺；更换 kernel、ROCm、
GPU 或路由表后应重新运行 benchmark。首次遇到新 packed offsets 时仍需校验并
构造 metadata，serving 性能依赖复用同一个不可变的 inference `cu_seqlens`，或
复用未被原地修改的普通 Tensor。

## 自动路由范围

Dense `auto` 只在 MI308X/gfx942、80 CU、BF16、K=V=128、T 为 64 的整数倍、
等长输入、FP32 state、无 decode prefix 的已验证范围内启用 Opus；packed
varlen `auto` 在相同硬件与 dtype/feature 约束下选择 WS，且各序列无需 64 对齐。
不满足条件时，所有参数和张量对象都会原样传给
`chunk_gated_delta_rule_opt_vk`，不会先做类型转换、连续化或输出分配。

对于 2026-07-31 closeout 的 477 个实测 shape，路由键为
`(B, T, H, state)`，并精确选择数据中的 CF/CS/WS/WF winner。不能把键压缩为
`(T, B*H, state)`：相同 `B*H` 的不同 B/H 分解已经出现过不同 winner。
只有 initial state 和 final state output 同时开启才对应 `state=on`；单边状态
I/O 没有测量数据，因此按 W/U 的固定 WS/WF 包络处理。

未命中 477 点表时，适配层使用代码内固定的 W/U 包络并强制选择 WS 或 WF，
不会调用读取 `OPUS_GDN_SPLIT_THRESHOLD` 的底层 auto mode。CF/CS 也通过显式
`c_mode` 选择，不依赖 `OPUS_GDN_K2C_SPLIT`。因此四路径 family 的选择是逐调用
确定的。统一入口还会关闭底层 backend 的 benchmark/debug 环境覆盖；即使模型
进程继承了 `OPUS_GDN_WF_VARIANT`、`OPUS_GDN_OUT_VARIANT`、
`OPUS_GDN_K2C_VARIANT` 等实验变量，也仍然执行 closeout 的发布配置。直接调用
W/U 或 C 底层 API 时，`use_env_overrides=True` 的默认值继续保留 A/B 能力。

## 数据来源

- 数据：`dense_closeout_all.json`
- SHA-256：`3e333710940d3c6d9feec6f8792fd500be5d0ac28423dc165ef80622cd712d89`
- 有效 shape：477
- winner：CF 196、CS 25、WS 182、WF 74
- 平台：AMD Instinct MI308X / gfx942 / 80 CU

路由数据以排序后的 32-bit packed records 固化在代码中，运行时不读取外部
JSON。内核或测量环境变化后，应重新跑完整矩阵并重新生成该表，不能按相邻
shape 静默插值。

## 当前边界

Dense 非 64 对齐输入仅在显式 Opus 路径中由 wrapper padding；dense auto 仍只
进入已验证的 64 对齐范围。Packed varlen 当前固定为 BT64 Neumann K1 + WS，
只支持等 head 数和 FP32 state；BF16 state、GQA/MQA、内核内 Q/K L2Norm 和
decode prefix 仍走现有实现。显式 Opus 路径若输入不受支持会直接报错，不会
悄悄换成另一条实现。
