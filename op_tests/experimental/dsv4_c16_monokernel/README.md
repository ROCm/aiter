# gfx1250 DeepSeek V4 Flash C16 SWA monokernel

Experimental kernel source snapshot, not an installed AITER operator or a
production ATOM integration. This preserves the last validated task-level
native-design implementation. It is slower than the native ATOM path.

## Scope and entry points

- DeepSeek V4 Flash, layer 0 SWA, concurrency 16, TP1, BF16 KV, 128-key window.
- Target serving workload: 1k input / 1k output. Results below are layer
  microbenchmarks, **not** serving TTFT, TPOT, or throughput measurements.
- Resident prefix: 256 CTAs, 128 threads / four wave32 waves per CTA.
- Default split: fused prefix -> native routed GU -> native routed GD -> fused
  reduction/add suffix. GU/GD use their native 3072-CTA launches.
- `work/swa-fourwave/schedule.py:build` constructs the full resident kernel.
- `work/swa-fourwave/split_moe.py:build` constructs the four-launch split path.
- `attention_pack.py` embeds four native one-wave attention tasks per CTA.
- `native_task_bodies.py`, `native_reduce_fold.py`, and
  `native_compiled_gemms.py` provide the native task adapters.
- `runtime.cpp` implements epoch publication/waits and early coefficient loads.

A stage supplies a FlyDSL device `body`, scalar ABI `types` and `values`,
logical task `total`, active `threads`, and `lds` bytes. The caller owns all
argument tensors and must keep them alive. The split builder additionally
expects the captured native GU/GD entries and the prepared output buffers.
These are internal research interfaces, not stable public APIs.

## Dependencies and workspace layout

Sources are preserved byte-for-byte with their historical `/work` paths.
Mount `work/swa-fourwave` at `/work/swa-fourwave`. Place the supplied bridge
and native GEMM support files under `/work/c16-native-reference` (or add them
into the existing research workspace). Do not overwrite an existing workspace
without reviewing its local changes.

Requires a gfx1250-capable ROCm/LLVM toolchain, FlyDSL, Gluon/Triton, AITER,
and the matching ATOM DeepSeek V4 implementation. AITER headers are expected
at `/app/aiter`. Current upstream dependency compatibility has not been tested.
`build_inputs.py` reconstructs LLVM bitcode from the supplied LLVM and HIP
sources without running a GPU kernel.

`prepare.py` and `bench_task_folded.py` retain the original checkpoint-backed
fixture interface. They additionally require the original research workspace's
`c16-native-reference/full_swa.py` fixture stack, `atom-actual-microbench`
helpers, `/app/ATOM`, `/data/DeepSeek-V4-Flash`, and locally prepared QKV/output
fixtures. Those model data and broader reference experiments are **not included**.
This is a kernel source handoff; the benchmark is not standalone in a fresh clone.

In that original fixture environment:

```bash
python build_inputs.py
cd /work/swa-fourwave
HIP_VISIBLE_DEVICES=0 TASK_NATIVE=1 NATIVE_PARITY=1 python bench_task_folded.py
HIP_VISIBLE_DEVICES=0 PROFILE_DIR=profiles-task-native python profile_split_bench.py
PROFILE_DIR=profiles-task-native python profile_analyze.py
```

Use separate Python processes for `TASK_NATIVE=0` and `TASK_NATIVE=1`; attention
specialization is selected at import. GPU0 was the only GPU used for validation.
The optional historical tagged-mHC experiments are not part of this handoff;
the default path has `TAGGED_MHC=0`.

## Last measured status

| Decode position | Previous split (us) | This split (us) | Native ATOM (us) |
|---|---:|---:|---:|
| 1024 | 225.52 | 263.15 | 169.26 |
| 1535 | 226.90 | 264.49 | 170.79 |
| 2047 | 225.99 | 263.66 | 169.74 |

The fully monolithic control measured 314-323 us. The latest split is about
55% slower than native. Its largest regression is attention body execution;
this is not established to be synchronization overhead. Prefix allocation:
642 VGPRs, no spills, one resident CTA per CU.

Historical validation passed 12 seed/position cases, bit-exact KV-cache
comparisons, and 150 graph stress replays. These GPU tests were not rerun when
publishing this snapshot. The source hash manifest identifies the published
files; the historical hash record identifies the measured kernel version.

See `work/swa-fourwave/NATIVE_TASK_AUDIT.txt` for all 31 operations and
`TASK_NATIVE_GAINS.txt` for instrumented gains/regressions. Paths to raw
artifacts in those historical reports refer to the original research workspace;
raw traces and benchmark datasets are not published here.

Two cooperative native bodies do not fit unchanged into four waves: initial
mHC's eight-wave apply/norm plus Sinkhorn, and routed quantization's eight-wave
LDS scale redistribution. They retain four-wave adaptations. Native GU/GD tasks
fit individually but remain external under the chosen split-MoE design.
