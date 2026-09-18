# Rebuilt tuned hipBLASLt versus FlyDSL B on gfx1250

Measured 2026-09-17 on the local gfx1250, 256-CU device.

This is the historical hipBLASLt tuning snapshot. It predates the final
row-major promotions for `512x7168x3072` and `512x2048x7168`; current
production defaults and the final split-K=1 native ASM comparison are recorded
in [GFX1250_FLYDSL_ASM_SCHED_ANALYSIS.md](GFX1250_FLYDSL_ASM_SCHED_ANALYSIS.md).

## Result

The final comparison rebuilt the hipBLASLt host library and regenerated its
device library from the merged six-shape logic. A pybind module calls the public
`hipblasLtMatmulAlgoGetHeuristic` and `hipblasLtMatmul` APIs. Both hipBLASLt and
FlyDSL B run in `test_gemm_a8w8_blockscale.py` with the same tensors, scales,
one hot buffer set, graph replay, two warmups, and 100 measured calls.

M is 512. Times are microseconds; lower is better.

| N | K | FlyDSL B | Rebuilt hipBLASLt | hipBLASLt / B | Result | Public index |
|---:|---:|---:|---:|---:|---:|---:|
| 6144 | 7168 | 14.331 | 15.531 | 1.084x | hipBLASLt 8.4% slower | 0 |
| 7168 | 3072 | 8.183 | 8.838 | 1.080x | hipBLASLt 8.0% slower | 1 |
| 7168 | 16384 | 24.511 | 29.150 | 1.189x | hipBLASLt 18.9% slower | 2 |
| 65536 | 1536 | 22.175 | 22.309 | 1.006x | hipBLASLt 0.6% slower | 3 |
| 2048 | 7168 | 9.507 | 11.094 | 1.167x | hipBLASLt 16.7% slower | 4 |
| 8192 | 1536 | 5.813 | 5.614 | 0.966x | hipBLASLt 3.4% faster | 5 |

The raw public-API numbers are in
[the rebuilt comparison CSV](hipblaslt_tuning/rebuilt_public_comparison.csv).
The public heuristic returned indices `0..5` in the order shown above, and each
reported solution name matches the corresponding solution in
[the merged logic](hipblaslt_tuning/gfx1250_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_UserArgs.yaml).

## Was hipBLASLt rebuilt after tuning?

Yes for the table above. The build has two distinct parts:

1. `TensileCreateLibrary` regenerated the msgpack lazy library and code objects
   from the merged six-shape tuning logic. Its log reports six parsed and six
   unique solutions.
2. hipBLASLt and `libtensilelite-host` were rebuilt together with
   `HIPBLASLT_ENABLE_YAML=OFF`, matching the generated msgpack library.

The earlier direct-artifact comparison did not require a hipBLASLt rebuild. It
loaded each generated Tensile library and launched its sole solution through
`SolutionAdapter`. Those results remain in
[the direct comparison CSV](hipblaslt_tuning/flydsl_comparison.csv) as a
secondary check. The direct and rebuilt public paths select the same six kernel
names and have similar timings.

The ROCm PyTorch wheel loads its bundled hipBLASLt by an explicit path before a
normal extension import. `LD_LIBRARY_PATH` alone therefore does not guarantee
that the pybind module binds to the rebuilt library. The public comparison uses
`LD_PRELOAD=<rebuilt-prefix>/lib/libhipblaslt.so.1`; an `LD_DEBUG=bindings`
check confirmed that the bridge's `hipblasLtMatmulAlgoGetHeuristic` and
`hipblasLtMatmul` symbols resolve to that rebuilt file.

## Native profile/config check

A second run enabled hipBLASLt's own profile logger with
`HIPBLASLT_LOG_MASK=64`; it did not use rocprofv3. The run used the same rebuilt
host library, generated device-library directory, Python benchmark, and input
configuration as the main comparison. The native log emitted one aggregated
entry per shape with 203 calls per entry.

| N | K | YAML `ExactLogic` index | Native profile index | Match | FlyDSL B (us) | Rebuilt hipBLASLt (us) |
|---:|---:|---:|---:|:---:|---:|---:|
| 6144 | 7168 | 0 | 0 | yes | 14.652 | 15.419 |
| 7168 | 3072 | 1 | 1 | yes | 7.896 | 8.797 |
| 7168 | 16384 | 2 | 2 | yes | 24.591 | 29.136 |
| 65536 | 1536 | 3 | 3 | yes | 21.768 | 21.938 |
| 2048 | 7168 | 4 | 4 | yes | 8.923 | 10.991 |
| 8192 | 1536 | 5 | 5 | yes | 5.634 | 5.871 |

The compact profile check is in
[the native profile CSV](hipblaslt_tuning/rebuilt_public_profile_check.csv).
The full solution name reported through the public API also equals
`Solutions[index].SolutionNameMin` for every row. These two checks confirm that
the rebuilt public API used all six entries from the merged config. The clean
timings above remain the primary performance result; this logged run is an
independent selection check.

The profile logger's `scaleA: 0, scaleB: 0` fields are incorrect for this MX
case. That logger derives the fields from the legacy `useScaleAB` string, which
is empty for block scaling. The matmul descriptor itself reads back
`HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` for both inputs. Its public API enum
value is `2`; the equivalent `hipblaslt-bench` argument value is
`--scaleA 3 --scaleB 3`. hipBLASLt maps API value `2` to its internal
`Block_32_UE8M0` value `3` before constructing the Tensile problem.

## Official client block-32 check

The official `hipblaslt-bench` client was built from hipBLASLt commit
`bc4ca6ea` and run against the rebuilt host library and merged gfx1250 device
library. Every invocation explicitly used `--scaleA 3 --scaleB 3`; the client
help identifies value `3` as `B32E8`, and every printed result row contains
`scaleA=3, scaleB=3`. The selected names also contain
`MXAE8B32_MXBE8B32`, and the merged logic declares `MXBlockA: 32` and
`MXBlockB: 32`.

| N | K | Client scale A | Client scale B | YAML index | Selected index | Match | Client time (us) |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 6144 | 7168 | 3 | 3 | 0 | 0 | yes | 16.2579 |
| 7168 | 3072 | 3 | 3 | 1 | 1 | yes | 9.65846 |
| 7168 | 16384 | 3 | 3 | 2 | 2 | yes | 30.1784 |
| 65536 | 1536 | 3 | 3 | 3 | 3 | yes | 23.6426 |
| 2048 | 7168 | 3 | 3 | 4 | 4 | yes | 12.5696 |
| 8192 | 1536 | 3 | 3 | 5 | 5 | yes | 6.72687 |

The parsed client output, including full solution names, is in
[the official block-32 CSV](hipblaslt_tuning/official_client_block32.csv).
These standalone-client timings use five cold calls and 100 timed calls with
the GPU timer. They verify mode and solution selection; the same-process
Python table above remains the direct performance comparison with FlyDSL B.

The client command for each `(N,K)` pair was:

```bash
env -u TENSILE_DB -u TENSILE_DB2 \
  LD_PRELOAD=<rebuilt-prefix>/lib/libhipblaslt.so.1 \
  HIPBLASLT_TENSILE_LIBPATH=<generated-library>/gfx1250 \
  hipblaslt-bench \
    -m 512 -n <N> -k <K> --transA T --transB N \
    --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r \
    --compute_type f32_r --scaleA 3 --scaleB 3 \
    --alpha 1 --beta 0 --algo_method heuristic --requested_solution 1 \
    --print_kernel_info --use_gpu_timer --initialization hpl \
    --cold_iters 5 --iters 100
```

## Why tuning did not broadly improve performance

Every tuning input contains one candidate and every run reports
`Actual Solutions: 1 / 1`. Running without `TENSILE_DB2` fixed the invalid
timing caused by its skip-kernel-launch bit, but it did not introduce other
kernels to search. The no-DB2 run selected the same full solution name as
`run1` for all six shapes.

The rebuild proves that the generated config is active; it does not create a
performance improvement by itself. Five shapes remain slower than FlyDSL B.
The `(8192,1536)` result is close enough that normal run-to-run variation can
flip the winner: hipBLASLt was 3.4% faster in this public-API run, while the
earlier direct run measured it 3.2% slower.

The generated standalone clients, run with `TENSILE_DB2` explicitly unset,
reported these valid kernel times:

| M | N | K | Generated-client time (us) |
|---:|---:|---:|---:|
| 512 | 6144 | 7168 | 17.1531 |
| 512 | 7168 | 3072 | 9.66048 |
| 512 | 7168 | 16384 | 31.8504 |
| 512 | 65536 | 1536 | 24.8367 |
| 512 | 2048 | 7168 | 12.6956 |
| 512 | 8192 | 1536 | 6.57079 |

The full names are in
[the tuning result CSV](hipblaslt_tuning/tuning_results.csv).
`TensileLogic --check-all` reports `Total 6 solutions`, `Keep 6 solutions`,
and `Reject 0 solutions` for the merged file.

## Output comparison

The rebuilt public hipBLASLt result is bitwise equal to FlyDSL B for three
shapes:

- `(7168,3072)`
- `(65536,1536)`
- `(8192,1536)`

For `(6144,7168)`, `(7168,16384)`, and `(2048,7168)`, FlyDSL B uses BF16
split-K partials while the selected Tensile solutions use FP32 partials or no
split-K. The fractions of differing BF16 output elements are `0.341667`,
`0.345369`, and `0.355141`. The FlyDSL speed advantage on those rows includes
that precision tradeoff.

## Reproduce the public comparison

Build the pybind bridge against the rebuilt hipBLASLt prefix:

```bash
ROCM_PATH=/path/to/rocm \
CXX=/path/to/rocm/bin/amdclang++ \
bash op_tests/build_hipblaslt_public_bridge.sh \
  /path/to/rebuilt/hipblaslt/prefix \
  /tmp/libhipblaslt_public_bridge.so
```

Before starting, ensure `fuser /dev/kfd /dev/dri/renderD*` prints no active
PIDs. Run one foreground process and leave `TENSILE_DB2` unset:

```bash
env -u TENSILE_DB2 \
  LD_PRELOAD=/path/to/rebuilt/hipblaslt/prefix/lib/libhipblaslt.so.1 \
  LD_LIBRARY_PATH=/path/to/rebuilt/hipblaslt/prefix/lib:/path/to/rocm/lib \
  HIPBLASLT_TENSILE_LIBPATH=/path/to/generated/library/gfx1250 \
  ENABLE_CK=0 GEMM_BENCH_ROTATE=1 GEMM_BENCH_GRAPH=1 \
  python3 op_tests/test_gemm_a8w8_blockscale.py \
    --flydsl --ck_preshuffle True \
    -m 512 \
    -nk 6144,7168 7168,3072 7168,16384 65536,1536 2048,7168 8192,1536 \
    --data-init uniform --scale-init auto --seed 0 \
    --hipblaslt-public-bridge /tmp/libhipblaslt_public_bridge.so \
    --table
```

The bridge performs heuristic selection once during construction, outside the
timed region. Scale packing and workspace allocation are also outside timing.
No profiler or background benchmark process is involved.
