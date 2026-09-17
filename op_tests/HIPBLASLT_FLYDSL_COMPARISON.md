# Rebuilt tuned hipBLASLt versus FlyDSL B on gfx1250

Measured 2026-09-17 on the local gfx1250, 256-CU device.

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
