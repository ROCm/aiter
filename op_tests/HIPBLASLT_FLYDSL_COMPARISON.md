# hipBLASLt/Tensile versus FlyDSL on gfx1250

Measured 2026-09-17 on the local gfx1250, 256-CU device.

## What was reproduced

`test_gemm_a8w8_blockscale.py` can now run the exact six Tensile winner
configurations from
[ROCm's FlyDSL comparison artifacts](https://github.com/ROCm/rocm-libraries/tree/users/pkamd/flydsl/projects/hipblaslt/tensilelite/flydsl_artifacts).
The checkout is pinned to `bc4ca6ea60abc3bd4f6980585a14f8bad742fcb2`.
The comparison calls these **hipBLASLt backend kernels through Tensile's native
solution adapter**, not the installed hipBLASLt public heuristic API. No separate
benchmark process is used to measure the winner.

Both paths use the test's actual tensors, `benchmark_data_init`, seed 0,
FP8 E4M3 uniform data, E8M0 `auto` scales, BF16 output, and the same `perftest`
implementation: 100 iterations, two warmups, matching buffer rotation and graph
settings. Both have explicit outputs, so outputs rotate along with inputs.
Default behavior without the new comparison option is preserved.

Tensile expects MX32 scales. Each activation block-128 scale is repeated four
times in K; each weight scale is repeated four times in K and 128 times in N.
The resulting **mathematical operands are identical**, not independently
generated approximations. The gfx1250 TN scale layout is
`[K/128, rows, 4]`. A CPU roundtrip checks this packing.

Packing/preshuffling is outside all timed regions. Tensile writes column-major
D; FlyDSL writes row-major D. The test uses a transpose *view* to compare
coordinates, without timing a layout conversion. These are GEMM-only results,
not an end-to-end row-major API comparison.

## Target: M=512, N=8192, K=1536

All times below are microseconds; lower is better.

| Timing regime | FlyDSL B-preshuffle | FlyDSL AB-preshuffle | Tensile winner |
|---|---:|---:|---:|
| Existing Python `perftest`, 100 rotating inputs/outputs, median of 3 runs | 8.202 | 7.043 | 5.932 |
| rocprof, same direct-launch regime, one run | 7.905 | 7.297 | 6.662 |
| rocprof, 100 rotating buffers, graph replay | 8.764 | 8.780 | 9.178 |
| rocprof, one hot buffer, graph replay | 5.717 | 5.900 | 5.728 |

The ordinary Python benchmark reproduces a Tensile lead (1.38x over B-preshuffle,
1.19x over AB-preshuffle). It is **not a sustained kernel-throughput lead**:
the lead disappears with matched graph replay, for both hot and rotating
buffers. Direct-launch traces show median gaps of **82.083 us / 82.323 us**
between FlyDSL B/AB kernels versus **4.366 us** between Tensile kernels.
GPU-only timing excludes those gaps from the reported duration, but launch
pacing can still change cache/memory-system state and clock behavior. We have
measured the pacing difference, not isolated every hardware mechanism behind it.

The rotating-graph durations are variable: standard deviations are
1.656 / 1.566 / 1.454 us. Hot-graph standard deviations are
0.523 / 0.511 / 0.563 us. Do not treat a few percent as a robust optimization.

For rocprof results, the summary uses 99 measured iterations after dropping the
first, excludes validation/warmup calls, and includes any split-K reduction
helper. Unlike `perftest`, it does not additionally remove IQR outliers.
Do not compare these methods as if their filtering and instrumentation were
identical. Nesting Torch's profiler with rocprof produced no usable CSV; the
external mode avoids that conflict.

## Six supplied shapes, matched hot-buffer graph replay

These are rocprof GPU-duration means, with reduction helpers included. M=512.

| N | K | FlyDSL B | FlyDSL AB | Tensile |
|---:|---:|---:|---:|---:|
| 6144 | 7168 | 14.366 | 13.916 | 15.512 |
| 7168 | 3072 | 8.475 | 7.880 | 8.704 |
| 8192 | 1536 | 5.717 | 5.900 | 5.728 |
| 2048 | 7168 | 9.353 | 9.385 | 11.036 |
| 65536 | 1536 | 21.836 | 22.255 | 22.238 |
| 7168 | 16384 | 25.003 | 24.886 | 28.704 |

The local current-software comparison does not reproduce a blanket hipBLASLt
lead. It is not an exact historical-container reproduction: the linked script
uses `rocm/fw-bringup:gfx1250-atom--flydsl-mxfp8-gemm-20260909`, and Docker is not
available here. The artifact directory contains configs/scripts but no published
performance logs. Its comparison parser reads `ck us`, not `apre us`, despite
enabling A-preshuffle in the command; its old text parser also does not accept
the current test's JSON summary. The Tensile configs use `KernelTime: false`,
1000 enqueues, hot buffers, and TrigSin/TrigCos input initialization, rather than
the current test's initialization/rotation/timing settings.

## Are the supplied tuning results used by hipBLASLt?

The installed stock hipBLASLt does **not** use the six supplied winners. The
artifact configs and `run_tuning.sh` write standalone Tensile build and timing
outputs; they do not merge the resulting exact mappings into the installed
hipBLASLt database or rebuild the library.

The requested command:

```bash
TENSILE_DB2=1 ./run_tuning.sh run1
```

cannot produce valid kernel timings. Bit 0 of `TENSILE_DB2` is implemented as
`Debug::skipKernelLaunch()`. Every enqueue printed `DEBUG: Skip kernel
execution`; the reported 0.16–1.34 us values were launch-skipping overhead and
the winner-validation passes failed.

Running the script again without `TENSILE_DB2` produced real timings:

| N | K | Tensile client time (us) |
|---:|---:|---:|
| 6144 | 7168 | 16.3235 |
| 7168 | 3072 | 10.0779 |
| 8192 | 1536 | 6.77071 |
| 2048 | 7168 | 12.5212 |
| 65536 | 1536 | 25.1202 |
| 7168 | 16384 | 30.1493 |

The 65536x1536 value is the initial `NO_CHECK` pass. Its later full-validation
pass was an anomalous 121.883 us. Each log says `Actual Solutions: 1 / 1`, so
this run recompiles and retimes the already specified winner. It does not search
for a better solution.

I then queried the public hipBLASLt heuristic using FP8 E4M3 A/B, BF16 D, FP32
accumulation, TN layouts, MX32 E8M0 scales, and a 128 MiB workspace limit. The
stock library returned the same index-193 `MT128x128x256` solution for every
shape. Its decompressed msgpack database contains that solution and none of the
three supplied winner tile signatures: `MT64x192x512`, `MT128x128x512`, or
`MT256x256x256`.

The six exact mappings were then merged into one gfx1250 logic file and hipBLASLt
was rebuilt. `TensileLogic --check-all` retained all six solutions and rejected
none. The public heuristic selected every expected winner. The rebuilt indices
below are local to this six-solution test library.

These are rocprof GPU-duration means for 100 hot-buffer graph calls after two
warmups. The GSU4 result includes its reduction helper.

| N | K | Stock selection | Stock (us) | Rebuilt selection | Rebuilt (us) | Speedup |
|---:|---:|---|---:|---|---:|---:|
| 6144 | 7168 | 193 / MT128x128x256 | 44.729 | 1 / MT64x192x512 | 14.010 | 3.19x |
| 7168 | 3072 | 193 / MT128x128x256 | 22.038 | 2 / MT128x128x512 | 8.356 | 2.64x |
| 8192 | 1536 | 193 / MT128x128x256 | 13.507 | 4 / MT128x128x512 | 5.826 | 2.32x |
| 2048 | 7168 | 193 / MT128x128x256 | 43.294 | 0 / MT128x128x512, GSU4 | 10.828 | 4.00x |
| 65536 | 1536 | 193 / MT128x128x256 | 96.456 | 5 / MT256x256x256 | 20.339 | 4.74x |
| 7168 | 16384 | 193 / MT128x128x256 | 95.714 | 3 / MT128x128x512 | 24.889 | 3.85x |

This confirms a large database-selection problem in the stock public path:
installing a library rebuilt with the supplied exact mappings improves these
shapes by 2.32x–4.74x. It does not establish that the supplied winners are the
best possible Tensile kernels, because the provided configs contain only one
candidate each. A real tuning pass must enumerate alternatives, merge its
winning logic into the production logic tree, and rebuild or redistribute the
hipBLASLt device library.

The public probe uses zero operands and constant scales to isolate selection and
kernel timing. The earlier same-process table uses matched random operands and
is the stronger FlyDSL-versus-winner comparison: there, the exact supplied
Tensile kernels remained slower on four of six shapes. The rebuilt public probe
therefore explains the much larger stock-library regression without erasing the
remaining kernel-level gap. The split-K precision qualification under
**Correctness and precision** still applies to the three larger-K FlyDSL
results.

The first custom build used `HIPBLASLT_ENABLE_YAML=ON`; in this checkout that
left `libhipblaslt.so` referring to msgpack loader symbols that the YAML-mode
`libtensilelite-host.so` did not provide. The working build uses the production
configuration, `HIPBLASLT_ENABLE_YAML=OFF` and
`HIPBLASLT_ENABLE_LAZY_LOAD=ON`, with `msgpack-cxx`. This loader issue is
independent of kernel performance.

## Correctness and precision

The target's constant-data case passes the reference check, and random outputs
are **bitwise identical** between both FlyDSL paths and Tensile. For random
data, all have the same existing 0.12187958% mismatch fraction against the test's
FP32 reference at `atol=rtol=0.01`; this is not a strict allclose pass.
Repeated native launches and graph replay must reproduce the initially
validated output exactly; the new harness checks this.

For (N,K)=(6144,7168), (2048,7168), and (7168,16384), the tuned FlyDSL paths use
split-K and **BF16 partial buffers**. The reduction adds in FP32, but cannot
undo that intermediate BF16 rounding. Tensile uses FP32 partials or no split.
The original test reports approximately 4.05%, 5.66%, and 4.29% reference
mismatches for FlyDSL, versus 0.239%, 0.223%, and 0.288% for Tensile.

A controlled check on 512x6144x7168 confirms the explanation: forcing FlyDSL
split-K=1 makes its output **bitwise identical to Tensile** and reduces the
reference mismatch fraction to 0.239%. Therefore the split-K speed advantages
in the table are not precision-equivalent.

## What can be ported to FlyDSL?

Both implementations already issue:

```asm
v_wmma_scale_f32_16x16x128_f8f6f4
```

It is wave32, FP8 E4M3 operands, E8M0 scaling, FP32 accumulation. There is no
different/faster WMMA opcode to import from the winner.

For the target, both use a 128x128 output tile, four waves, and a 4x4 cluster.
Tensile uses `DepthU=512`, two LDS buffers, `PrefetchGlobalRead=2`,
`PrefetchGL2=1`, `ScheduleIterAlg=4`, segmented/padded LDS, TDM loads, and
`matrix_a_reuse` hints. FlyDSL's name says K=128, but its `KPAIR=2` means
**256 K elements per TDM stage**, with four buffers. Both therefore stage about
1024 K elements, so this is not simply "Tensile has a 4x larger prefetch window."

| Target resource | FlyDSL B | Tensile |
|---|---:|---:|
| Actual LDS bytes | 271424 (dynamic) | 272384 (fixed) |
| ISA metadata VGPR count | 392 | 1022 |
| rocprof VGPR count | 200 | 512 |
| Scratch / VGPR spills | 0 / 0 | 0 / 0 |

The profiler reports zero *fixed* LDS for FlyDSL; that does not mean it uses no
LDS. These gfx1250 VGPR representations differ; CDNA3 occupancy formulas must
not be applied to them. Tensile's deeper schedule also has substantially more
register allocation, so transplanting it wholesale is not automatically better.

Scoped port candidates:

1. **Operand-reuse hints and WMMA order:** `WMMAScale(reuse_a=..., reuse_b=...)`
   is already supported by installed FlyDSL. The current kernel does not set
   those hints. Match hints to actual operand lifetimes/order and verify final
   ISA; this is a small, feasible experiment, not a proven speedup.
2. **L2 prefetch:** FlyDSL already exposes `rocdl.tdm_ops.l2_prefetch_tile`.
   Tensile uses `global_prefetch_b8`. Tune the distance against cold/rotating
   workloads; do not assume it helps the short K=1536 case. A generated
   target-only ablation changing just `PrefetchGL2: [1]` to `[0]` gave a
   three-run direct-perftest median of 6.042 us versus 5.932 us with prefetch.
   The run ranges overlap; this small difference does not explain the large
   direct-benchmark lead. Outputs remained bitwise identical. The ablation
   config, code object, and logs are in `/tmp/hipblaslt_noprefetch.vRR7ZH`.
3. **512-K TDM grouping and LDS scheduling:** expressible with existing TDM,
   LDS padding, and scheduling primitives, but requires coordinated layout,
   scale-address, tail, barrier, and register-liveness changes. Changing
   `tile_k` alone is not a valid port of this schedule.
4. **FP32 split-K partials:** needed for a precision-matched comparison on the
   larger-K cases. This increases scratch/LDS traffic and needs retuning.
5. **Launch path:** capture/replay graphs or reduce FlyDSL dispatch work when
   launch-bound. The measured 82-us versus 4-us dispatch gaps are not something
   a WMMA tile change fixes.

No production FlyDSL kernel was changed by this comparison. The earlier
target-only cluster tuning remains. The previously tested 96x32 candidate was
about 14.54 us versus about 6.7 us for the then-tested 128x128 baseline, under
that earlier benchmark regime. Neither supplied winner evidence nor those
measurements support switching this target to 96x32.

## Run in the current workspace

The existing generated winners and bridge are under
`/tmp/hipblaslt_flydsl_repro.ZX9FzA`.

```bash
ENABLE_CK=0 GEMM_BENCH_ROTATE=100 \
python3 op_tests/test_gemm_a8w8_blockscale.py \
  --flydsl --ck_preshuffle True --apre True \
  -m 512 -nk 8192,1536 \
  --data-init uniform --scale-init auto --seed 0 \
  --hipblaslt-winner-dir /tmp/hipblaslt_flydsl_repro.ZX9FzA/winner34 \
  --hipblaslt-bridge /tmp/hipblaslt_flydsl_repro.ZX9FzA/libwinner_bridge.so
```

Use `GEMM_BENCH_GRAPH=1` for graph replay. `GEMM_BENCH_ROTATE=1` reuses one
buffer set; `100` rotates 100 sets. Default `0` retains AITER's automatic
rotation selection. For all six shapes, point `--hipblaslt-winner-dir` at the
parent reproduction directory and pass the six `-nk` pairs shown above.

For independent GPU timing, prepend:

```bash
ENABLE_CK=0 GEMM_BENCH_ROTATE=100 GEMM_BENCH_GRAPH=1 GEMM_BENCH_EXTERNAL=1 \
rocprofv3 --stats --kernel-trace -f csv -d /tmp/my_gemm_trace -o compare -- \
python3 op_tests/test_gemm_a8w8_blockscale.py ...
```

`GEMM_BENCH_EXTERNAL=1` runs the common warmup/rotation loop without Torch's
profiler. Its Python timing fields are deliberately NaN; read rocprof's CSV.

## Build notes and artifacts

The native bridge source is `op_tests/csrc/hipblaslt_winner_bridge.cc`, with
`op_tests/build_hipblaslt_winner_bridge.sh` as its build entry point. First build
the pinned upstream checkout's `tensilelite` CMake preset. This environment has
split ROCm packaging; the successful configuration used:

```bash
cmake --preset tensilelite \
  -S /tmp/hipblaslt_flydsl_repro.ZX9FzA/rocm-libraries/projects/hipblaslt \
  -B /tmp/hipblaslt_flydsl_repro.ZX9FzA/build -G Ninja \
  -DGPU_TARGETS=gfx1250 \
  -DCMAKE_C_COMPILER=/usr/local/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/usr/local/bin/amdclang++ \
  '-DCMAKE_PREFIX_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel;/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib/llvm;/opt/rocm' \
  -DPython_EXECUTABLE=/tmp/hipblaslt_flydsl_repro.ZX9FzA/venv/bin/python3 \
  -DPython3_EXECUTABLE=/tmp/hipblaslt_flydsl_repro.ZX9FzA/venv/bin/python3 \
  -DTENSILELITE_ENABLE_AUTOBUILD=OFF \
  '-DCMAKE_CXX_FLAGS=--hip-path=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel -Wno-unused-command-line-argument'
cmake --build /tmp/hipblaslt_flydsl_repro.ZX9FzA/build --parallel 24

ROCM_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel \
bash op_tests/build_hipblaslt_winner_bridge.sh \
  /tmp/hipblaslt_flydsl_repro.ZX9FzA/rocm-libraries \
  /tmp/hipblaslt_flydsl_repro.ZX9FzA/build \
  /tmp/hipblaslt_flydsl_repro.ZX9FzA/libwinner_bridge.so
```

`HIP_HAS_CLUSTER_LAUNCH` must pass. Initial configuration without `--hip-path`
falsely failed the link test because the core ROCm package lacks the unversioned
HIP library; no system symlinks were changed. Python dependencies and build logs
are retained in the reproduction directory. The setup used an isolated Python
environment for Tensile generation; the actual comparison uses system Torch.

`generate_winners.sh 30 31 32 33 34 35` in the reproduction directory runs
Tensile `--build-only` on the supplied configs using the prebuilt client and
rocisa module. Each winner directory contains `ClientParameters.ini`,
`TensileLibrary.yaml`, `.co`, and the generated `.s` assembly.

Authoritative matched-output artifacts in that directory:

- `target_matched_r100_{1,2,3}.log`: repeated existing-perftest target results.
- `six_matched_rotation.log`, `six_matched_hotgraph.log`: same-file comparisons.
- `profile_matched_direct/`: independent direct-launch target trace.
- `profile_matched_hotgraph/`, `profile_matched_rotategraph/`: six-shape traces.
- `summarize_traces.py`: excludes warmups/validation, sums reduction helpers;
  resulting `summary.json` files include mean, median, min, max, and standard
  deviation.
- `check_splitk_accuracy.py`, `splitk_accuracy.log`: precision isolation check.
- `final_verification.log`: rebuilt bridge, constant and random initialization,
  bitwise B/AB/Tensile comparisons and repeated graph-output checks.

Earlier `same_python*`, `six_same_python*`, and `profile_six*` artifacts predate
matched output rotation and are **not** the final comparison.

The public heuristic probe is `op_tests/csrc/hipblaslt_public_bench.cc`, built
with `op_tests/build_hipblaslt_public_bench.sh`. For example:

```bash
ROCM_PATH=/path/to/rocm \
bash op_tests/build_hipblaslt_public_bench.sh /path/to/hipblaslt-prefix \
  /tmp/hipblaslt_public_bench

HIPBLASLT_TENSILE_LIBPATH=/path/to/hipblaslt/library/gfx1250 \
LD_LIBRARY_PATH=/path/to/hipblaslt/lib:/path/to/rocm/lib \
/tmp/hipblaslt_public_bench 512 8192 1536 100 1
```

The stock and rebuilt public traces and their parsed summary are under
`/tmp/hipblaslt_public_profiles`. The working custom msgpack build and install
trees are `/tmp/hipblaslt_six_msgpack/release2` and
`/tmp/hipblaslt_six_msgpack/install2`.
