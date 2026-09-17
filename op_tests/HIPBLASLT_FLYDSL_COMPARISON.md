# Direct `run1` Tensile winners versus FlyDSL B on gfx1250

Measured 2026-09-17 on the local gfx1250, 256-CU device.

## Result

The comparison now loads the exact generated `run1` Tensile library and code
object for each shape. It does not call the installed hipBLASLt heuristic and
does not use a separate profiler. Both implementations run in
`test_gemm_a8w8_blockscale.py` with its normal Python `perftest` path, the same
input tensors, the same scales, one hot buffer set, graph replay, two warmups,
and 100 measured calls.

M is 512. Times are microseconds; lower is better.

| N | K | FlyDSL B | Exact `run1` winner | Winner / B | Result |
|---:|---:|---:|---:|---:|---:|
| 6144 | 7168 | 14.213 | 15.481 | 1.089x | winner 8.9% slower |
| 7168 | 3072 | 8.259 | 8.844 | 1.071x | winner 7.1% slower |
| 7168 | 16384 | 24.500 | 28.940 | 1.181x | winner 18.1% slower |
| 65536 | 1536 | 22.134 | 22.038 | 0.996x | winner 0.4% faster |
| 2048 | 7168 | 9.377 | 10.696 | 1.141x | winner 14.1% slower |
| 8192 | 1536 | 5.663 | 5.890 | 1.040x | winner 4.0% slower |

This is the fresh pybind run after the device recovered. Each shape prints
`hipBLASLt/Tensile explicit winner` followed by the full solution name, which
confirms that the generated solution is loaded directly.

## Are the tuning results used?

Yes, in this comparison. The bridge searches only under the supplied `run1`
directory, matches the shape through `ClientParameters.ini`, loads that entry's
`TensileLibrary.yaml`, loads the adjacent generated code object, and launches
the library's single solution through `TensileLite::hip::SolutionAdapter`.
There is no heuristic lookup that could substitute a stock solution.

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
times of 6.77-30.15 us across these shapes. Each client reported
`Actual Solutions: 1 / 1`: these artifact configs time one prescribed solution;
they do not search a wider candidate set. The label `winner` therefore means
the solution supplied by the artifact, not proof that a full local tuning search
found the fastest possible Tensile kernel.

### Validated 8192x1536 rerun without `TENSILE_DB2`

The `flydsl_mxf8_tn_34_winner.yaml` config was rerun by itself with
`TENSILE_DB2` explicitly unset. The client launched the kernel, validation
passed, and the measured client time was **6.57079 us**. The output is under:

```text
/tmp/hipblaslt_flydsl_repro.ZX9FzA/rocm-libraries/projects/hipblaslt/
tensilelite/flydsl_artifacts/run_8192_1536_no_db2
```

The rerun still reported `Actual Solutions: 1 / 1` and selected the same
`MT128x128x512` solution as `run1`. A direct same-Python comparison using the
newly generated directory measured 5.769 us for FlyDSL B and 5.503 us for the
Tensile winner, with bitwise-equal outputs. An earlier run of the identical
solution measured 5.663 us and 5.890 us respectively, so the roughly 4-5%
ordering at this short shape is within observed run-to-run variation.

## Why the exact winners are slower

The direct results rule out the original database-selection hypothesis for this
comparison. The requested `run1` solution is active for every row, and FlyDSL B
is still faster under the shared Python timing setup.

Three rows are also not precision-equivalent. FlyDSL B uses BF16 split-K
partials for `(N,K)=(6144,7168)`, `(2048,7168)`, and `(7168,16384)`, while the
Tensile winners use FP32 partials or no split-K. The fractions of output elements
that differ between the two implementations are 0.341667, 0.355141, and
0.345369 respectively. FlyDSL's 8.9-18.1% advantage on those rows includes
that lower-precision split-K tradeoff.

The other three rows are bitwise equal:

- `(7168,3072)`: winner is 7.1% slower.
- `(65536,1536)`: winner is 0.4% faster, effectively tied at this noise level.
- `(8192,1536)`: winner is 4.0% slower.

For those rows, the remaining gap is kernel performance and normal run-to-run
variation. It is not caused by the tuned solution being skipped or replaced.

## Reproduce in the same Python script

```bash
ENABLE_CK=0 GEMM_BENCH_ROTATE=1 GEMM_BENCH_GRAPH=1 \
python3 op_tests/test_gemm_a8w8_blockscale.py \
  --flydsl --ck_preshuffle True \
  -m 512 \
  -nk 6144,7168 7168,3072 7168,16384 65536,1536 2048,7168 8192,1536 \
  --data-init uniform --scale-init auto --seed 0 \
  --hipblaslt-winner-dir \
    /tmp/hipblaslt_flydsl_repro.ZX9FzA/rocm-libraries/projects/hipblaslt/tensilelite/flydsl_artifacts/run1 \
  --hipblaslt-bridge \
    /tmp/hipblaslt_flydsl_repro.ZX9FzA/libwinner_bridge.so
```

Do not pass `--apre True`; this comparison is FlyDSL B only.

## Direct pybind bridge

`hipblaslt_winner_bridge.cc` exposes the generated winner through pybind. The
Python wrapper passes the existing Torch tensor addresses and current Torch
stream directly into the Tensile solution adapter. Scale packing and allocation
remain outside the timed calls.

Build it against the same checkout used to generate `run1`:

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
