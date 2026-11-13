# Opt-in BF16 HGEMM with FP32 workspace

`aiter.ops.flydsl.flydsl_hgemm_fp32_workspace` is an explicit inference entry
for `(M,N,K) = (128,7168,12288)` on **gfx950 with 256 CUs**.
A is BF16 `[128,12288]`, weight is BF16 `[7168,12288]`, and the final output
is BF16 `[128,7168]`. The shape was supplied as a standalone GEMM workload,
not inferred from a model or parallelism configuration.

## What runs

1. The existing FlyDSL GEMM writes FP32 with tile `128x128x64`,
   stages 4, split-K 4, waves `2x4x1`, group-M 0, K-stagger 0 and
   B non-temporal loads.
2. PyTorch copies/converts FP32 to the caller's BF16 output.

This is **two kernels**, not a fused epilogue or a new device kernel.
The entire chain is timed. Inputs and weights are not quantized or prepacked.
FP32 atomic split-K avoids BF16 rounding at each atomic accumulation, but
does not guarantee bitwise deterministic summation. Deterministic-algorithms
mode is rejected.

Normal HGEMM dispatch, tuned CSVs and their existing numerics do not change.
There is no automatic selection or model integration.
The runtime default for this shape is a PyTorch fallback, not the previously
tested FlyDSL kernel used as an optimization comparator below.

## Calling and ownership

    import torch
    from aiter.ops.flydsl import flydsl_hgemm_fp32_workspace

    out = torch.empty((128, 7168), device=a.device, dtype=torch.bfloat16)
    workspace = torch.empty((128, 7168), device=a.device, dtype=torch.float32)
    flydsl_hgemm_fp32_workspace(a, weight, out, workspace)

The caller-provided FP32 workspace is **3.5 MiB**. Neither output nor
workspace may overlap either input or each other. Scratch need not be
initialized by the caller: the existing GEMM initializes it per invocation.
All tensors must be 16-byte aligned, contiguous row-major, on one device and
without autograd or lazy conjugate/negative metadata. Bias, scaling and
prepacked weights are unsupported.

Sequential calls on one stream can reuse scratch. Concurrent streams or graph
replays require separate scratch/output allocations or explicit dependencies.
When passing `stream=`, order producers and consumers and retain all tensors
until work completes. Warm up before graph capture and keep buffers alive for
replay. There is no host tensor-content read or per-call output/scratch
allocation in this wrapper; first-use backend/JIT initialization remains
the responsibility of warmup.

## Performance and limitations

Hardware: MI355X, gfx950 / 256 CU. Submitted-entry runtime:
PyTorch `2.10.0+rocm7.2.4.git3d3aa833`, HIP `7.2.53211`, pinned FlyDSL `0.5.0`.
AITER upstream base: `5ff09ac`. The underlying gfx950 HGEMM device source is
unchanged from the earlier experiments.

Seed 7202, same-card adjacent ABBA/BAAB, 12 blocks, 128 complete operations per
graph replay, medians in microseconds:

| Comparison | Mode | Comparator | FP32-workspace chain |
|---|---|---:|---:|
| Previous tested FlyDSL split1 | 16 independent A/W/output sets | 63.843 | 48.576 |
| hipBLASLt solution 440342 | 16 independent A/W/output sets | 47.879 | 48.537 |
| Previous tested FlyDSL split1 | One hot A/W/output set | 40.115 | 49.394 |
| hipBLASLt solution 440342 | One hot A/W/output set | 38.985 | 49.443 |

The chain is about 24% lower latency than the previous FlyDSL split1 for
rotating weights, and about 1.4% slower than this tuned hipBLASLt solution.
**It regresses hot-weight performance**, so it is opt-in rather than a new
default. hipBLASLt solution IDs are library-version-specific.

Seed 7201 was affected by co-tenancy, with large within-block variation and
unreliable A/A controls. Both seeds are retained in
[hgemm_fp32_workspace_samples.jsonl](../op_tests/op_benchmarks/flydsl/hgemm_fp32_workspace_samples.jsonl):
all comparisons and blocks, no outlier removal. The low-variation seed does
not make the shared node exclusive. These are kernel-chain microbenchmarks,
not model or end-to-end throughput claims.

Rotation reduces cross-call weight reuse, but does not guarantee zero cache
hits or measure only HBM access time. TB/s uses logical A/W/output bytes,
not hardware-measured HBM traffic or a guessed scratch-atomic traffic count.

## Reproduce

Use an available GPU and the pinned dependencies.

    python op_tests/test_flydsl_hgemm.py -s 128,7168,12288 --rotate 1 16 --seed 7102

    python op_tests/op_benchmarks/flydsl/bench_hgemm_fp32_workspace.py \
      --rotate 16 --nodes 128 --rounds 12 --seed 7202 \
      --hip-solutions 440342 440676 440108 --trace --json workspace.json

Omit `--hip-solutions` to compare against the previous FlyDSL split1 without
depending on local hipBLASLt IDs. Unsupported solutions and numerical
failures abort rather than silently fall back.

The op test covers metadata/overlap/determinism/prepacked-weight rejection,
output/workspace canaries, two changed A/W sets after capture, all rotating
outputs, zero and each K-quarter contribution, independent stream graphs and
an explicit non-current stream. It checks `rtol=atol=0.01`, **all elements**,
against FP32 reference computation rounded to BF16. Both formal benchmark
seeds also check all rotating outputs.

Separate eager diagnostics recorded 8 FP32 GEMMs and 8 conversion kernels for
8 calls; those traces are not used to derive graph timing.
No full-domain numerical proof, torch.compile/AOT deployment, end-to-end model
validation or other-architecture performance qualification is claimed.
