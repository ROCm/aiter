# Direct no-DB2 Tensile artifacts versus FlyDSL B on gfx1250

Measured 2026-09-17 on the local gfx1250, 256-CU device.

## Result

The comparison loads the exact freshly generated Tensile library and code
object for each shape. It does not call the installed hipBLASLt heuristic or
use a separate profiler. Both implementations run in
`test_gemm_a8w8_blockscale.py` with its normal Python `perftest` path, the same
input tensors, the same scales, one hot buffer set, graph replay, two warmups,
and 100 measured calls.

M is 512. Times are microseconds; lower is better.

| N | K | FlyDSL B | Exact Tensile solution | Tensile / B | Result |
|---:|---:|---:|---:|---:|---:|
| 6144 | 7168 | 14.355 | 15.516 | 1.081x | Tensile 8.1% slower |
| 7168 | 3072 | 8.455 | 8.799 | 1.041x | Tensile 4.1% slower |
| 7168 | 16384 | 24.921 | 28.959 | 1.162x | Tensile 16.2% slower |
| 65536 | 1536 | 21.858 | 22.199 | 1.016x | Tensile 1.6% slower |
| 2048 | 7168 | 8.910 | 11.012 | 1.236x | Tensile 23.6% slower |
| 8192 | 1536 | 5.724 | 5.905 | 1.032x | Tensile 3.2% slower |

The raw numbers are in
[the same-script comparison CSV](hipblaslt_tuning/flydsl_comparison.csv). Each
shape printed `hipBLASLt/Tensile explicit winner` followed by the full solution
name, confirming that the newly generated solution was loaded directly.

## Are the tuning results used?

Yes, in this comparison. The bridge searches only the index of the newly
generated no-DB2 artifacts, matches the shape through `ClientParameters.ini`,
loads that entry's `TensileLibrary.yaml`, loads the adjacent generated code
object, and launches the library's single solution through
`TensileLite::hip::SolutionAdapter`. There is no heuristic lookup that could
substitute a stock solution.

The values in `run1/tuning_results.csv` are not valid performance numbers,
however. The run used:

```bash
TENSILE_DB2=1 ./run_tuning.sh run1
```

Bit 0 of `TENSILE_DB2` enables TensileLite's skip-kernel-launch debug mode. The
client logged `DEBUG: Skip kernel execution`, produced unrealistically small
0.16-1.34 us values, and failed winner validation. The generated YAML and code
objects can still be launched, which is what the table above measures.

Running the generated clients without `TENSILE_DB2` gave real standalone-client
times of 6.57-31.85 us across these shapes. Each client reported
`Actual Solutions: 1 / 1`: these artifact configs time one prescribed solution;
they do not search a wider candidate set. The label `winner` therefore means
the solution supplied by the artifact, not proof that a full local tuning search
found the fastest possible Tensile kernel.

### Validated six-shape rerun without `TENSILE_DB2`

All six artifact configs were rerun individually with `TENSILE_DB2` explicitly
unset. Every generated client launched its kernel and passed validation.

| M | N | K | Generated-client time (us) |
|---:|---:|---:|---:|
| 512 | 6144 | 7168 | 17.1531 |
| 512 | 7168 | 3072 | 9.66048 |
| 512 | 7168 | 16384 | 31.8504 |
| 512 | 65536 | 1536 | 24.8367 |
| 512 | 2048 | 7168 | 12.6956 |
| 512 | 8192 | 1536 | 6.57079 |

The complete solution names and times are in
[the combined tuning CSV](hipblaslt_tuning/tuning_results.csv). The six
one-shape outputs were merged into
[one gfx1250 library-logic config](hipblaslt_tuning/gfx1250_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_UserArgs.yaml).
It contains six solutions with indices `0..5` and six corresponding
`ExactLogic` entries. `TensileLogic --check-all` reports `Total 6 solutions`,
`Keep 6 solutions`, and `Reject 0 solutions`.

The raw generated outputs are under:

```text
/tmp/hipblaslt_flydsl_repro.ZX9FzA/rocm-libraries/projects/hipblaslt/
tensilelite/flydsl_artifacts/run_other5_no_db2
/tmp/hipblaslt_flydsl_repro.ZX9FzA/rocm-libraries/projects/hipblaslt/
tensilelite/flydsl_artifacts/run_8192_1536_no_db2
```

Each rerun reported `Actual Solutions: 1 / 1`. These runs validate the
prescribed artifact solutions; the inputs do not provide a wider candidate set
for a full tuning search.

## Why tuning did not improve the result

The no-DB2 generation selected the same full solution name as `run1` for every
shape. Every input reported `Actual Solutions: 1 / 1`, so `run_tuning.sh` had
no alternative kernels to compare. Unsetting `TENSILE_DB2` made the generated
client launch and measure the kernel correctly, but it did not expand the
candidate set or choose a different implementation.

The direct Tensile times before and after regeneration are also effectively
the same. A negative change means the new measurement was faster.

| N | K | Previous direct run (us) | New no-DB2 artifact (us) | Change |
|---:|---:|---:|---:|---:|
| 6144 | 7168 | 15.481 | 15.516 | +0.2% |
| 7168 | 3072 | 8.844 | 8.799 | -0.5% |
| 7168 | 16384 | 28.940 | 28.959 | +0.1% |
| 65536 | 1536 | 22.038 | 22.199 | +0.7% |
| 2048 | 7168 | 10.696 | 11.012 | +3.0% |
| 8192 | 1536 | 5.890 | 5.905 | +0.3% |

## Why the exact winners are slower

The direct results rule out the original database-selection hypothesis for this
comparison. The newly generated solution is active for every row, and FlyDSL B
is faster under the shared Python timing setup.

Three rows are also not precision-equivalent. FlyDSL B uses BF16 split-K
partials for `(N,K)=(6144,7168)`, `(2048,7168)`, and `(7168,16384)`, while the
Tensile solutions use FP32 partials or no split-K. The fractions of output
elements that differ between the two implementations are 0.341667, 0.355141,
and 0.345369 respectively. FlyDSL's 8.1-23.6% advantage on those rows includes
that lower-precision split-K tradeoff.

The other three rows are bitwise equal:

- `(7168,3072)`: Tensile is 4.1% slower.
- `(65536,1536)`: Tensile is 1.6% slower.
- `(8192,1536)`: Tensile is 3.2% slower.

For those rows, the remaining gap is kernel performance and normal run-to-run
variation. It is not caused by the tuned solution being skipped or replaced.

## Reproduce in the same Python script

```bash
env -u TENSILE_DB2 ENABLE_CK=0 GEMM_BENCH_ROTATE=1 GEMM_BENCH_GRAPH=1 \
python3 op_tests/test_gemm_a8w8_blockscale.py \
  --flydsl --ck_preshuffle True \
  -m 512 \
  -nk 6144,7168 7168,3072 7168,16384 65536,1536 2048,7168 8192,1536 \
  --data-init uniform --scale-init auto --seed 0 \
  --hipblaslt-winner-dir \
    /tmp/hipblaslt_flydsl_repro.ZX9FzA/rocm-libraries/projects/hipblaslt/tensilelite/flydsl_artifacts/run_all6_no_db2_index_v2 \
  --hipblaslt-bridge \
    /tmp/hipblaslt_flydsl_repro.ZX9FzA/libwinner_bridge.so
```

Do not pass `--apre True`; this comparison is FlyDSL B only.

## Direct pybind bridge

`hipblaslt_winner_bridge.cc` exposes the generated winner through pybind. The
Python wrapper passes the existing Torch tensor addresses and current Torch
stream directly into the Tensile solution adapter. Scale packing and allocation
remain outside the timed calls.

Build it against the same checkout used to generate the Tensile artifacts:

```bash
CXX=/usr/local/bin/amdclang++ \
ROCM_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel \
PYTHON=python3 \
bash op_tests/build_hipblaslt_winner_bridge.sh \
  /tmp/hipblaslt_flydsl_repro.ZX9FzA/rocm-libraries \
  /tmp/hipblaslt_flydsl_repro.ZX9FzA/build \
  /tmp/hipblaslt_flydsl_repro.ZX9FzA/libwinner_bridge.so
```

The bridge build and the complete six-shape Python comparison both succeed with
the current checkout. Before another GPU run, check that no Python, ROCm query,
or profiler process is already using the device; this machine can hang when
multiple GPU processes overlap.
