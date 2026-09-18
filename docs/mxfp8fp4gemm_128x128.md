# gfx1250 MXFP8 GEMM 128×128 K128/PF8

The `(M,N,K)=(512,8192,1536)` FP8×FP8 case selects a 128×128 K128/PF8
kernel for both `a_preshuffle=True` (AP1) and `False` (AP0). Other shapes retain
the existing tile preferences. If the 128×128 CSV row for the requested A layout
is absent, this shape falls back to 256×256. Output is BF16.

## Kernel configuration

| Setting | AP1 | AP0 |
|---|---|---|
| Output tile / K step / prefetch stages | 128×128 / 128 / 8 | Same |
| A / B preshuffle | 1 / 1 | 0 / 1 |
| Cluster X×Y / traversal grid X×Y | 4×4 / 8×2 | Same |
| Persistent workgroups / threads per workgroup | 256 / 128 | Same |
| Input early timeout / output descriptor ET | 1 / 1 | Same |
| A and ScaleA | HT/CU | Same |
| B and ScaleB | RT_NT/CU | Same |
| D | BYPASS/SYS | NT/SYS |
| LDS / VGPR allocation | 278528 bytes / 896 | 286720 bytes / 896 |
| Packed kernargs / preload | 80 bytes / 20 dwords | Same |
| Split-K capability | None; `splitk=1` only | Same |

AP0 preserves the POC variant validated on 2026-09-11 (19/19 CPU golden checks)
and its NT/SYS output policy. Its assembly was imported with only the entry
symbol renamed and trailing whitespace removed. It is a separate layout and
code object; passing row-major A to the AP1 kernel is not supported.

The CSV `splitk=0` is a capability flag, not the runtime split count.
The public Python API resolves automatic split-K to 1 for this variant and
returns the GEMM output directly without `torch.sum`.

Explicit AP1 selection requires positive M/N multiples of 128; AP0 requires
positive M/N multiples of 512 (complete 4×4 clusters). Both require K≥1024 and
K%128=0. Eight K stages are preloaded; shorter K and partial output
tiles have not been enabled. The imported AP0 variant faulted on the
128×128×1024 small-grid check, so its stricter guard rejects that explicit
selection before launch. FP4 keeps its existing variants. No generalized
performance preference is inferred from the indexer shape.

## AP1 validation and performance (2026-09-16)

Measured on gfx1250 on 2026-09-16 for `(512,8192,1536)`, using the native
op test with POC input/scale initialization, split-K=1, 50 warmups,
200 profiled iterations per run, 30 input buffers and one output buffer:

| Kernel | Three runs (us) | Minimum (us) | Median (us) |
|---|---|---:|---:|
| 128x128 K128/PF8 | 5.69, 5.67, 5.61 | 5.61 | 5.67 |
| 256x256 comparison | 8.13, 8.23, 7.99 | 7.99 | 8.13 |

Each run uses the existing profiler timing: discard the first event and
apply IQR filtering before averaging. The median above is across three runs.
Runs started after the GPU was idle and passed process and temperature checks.

CPU dispatch/fallback checks, shape and split-K guards, HIP compilation,
and POC/aiter instruction comparison passed. GPU checks passed for the target
POC input through automatic dispatch, explicit selection and the public API.

The full integration suite has **not** passed: the `uniform/auto` random-input
check reported 4 of 4,194,304 elements outside the unchanged `rtol=0.1`,
`atol=1.0` tolerance (maximum absolute difference 5.75). The cause is unresolved.
Minimum-K, persistent-tail and small-grid GPU checks were not reached after
that warning. The timings above do not certify correctness on those inputs.

## AP0 integration validation (2026-09-18)

The registered AP0 `.text` matches the original POC code object byte for byte.
The actual C++ selector passed CPU checks for AP0/AP1 selection, the existing
five 256×256 benchmark shapes, dtype/layout separation, split-K limits, shape
guards and fallback when the AP0 row is absent.

Using a fresh JIT directory on gfx1250:

- The indexer case passed constant/constant and POC data/scale checks through
  automatic dispatch. Automatic and explicit public Python API calls also passed.
- All six AP0 cases passed constant/constant checks. Uniform/auto retained the
  existing sparse warnings at unchanged `rtol=0.1`, `atol=1.0`; the indexer case
  had 4 of 4,194,304 elements outside tolerance, maximum absolute difference 5.75.
- Sixteen explicit AP0 checks passed using POC data/scales: 512×512 with
  K=1024..1920 in steps of 128, and 2048×4096 with K=2048..2944 in steps of 128.
  These cover all eight K ring exits and persistent transitions.
- The perf wrapper passed an AP0 indexer smoke run with both input pairs and two
  repeats each, plus an AP1 constant-input regression run. Every accepted formal
  call contained 100 records for the intended 128×128 kernel, and the terminal,
  CSV and JSON identified the selected A layout.
- After aligning the prebenchmark order with FlyDSL, both AP0 and AP1 passed
  two repeats of constant and uniform inputs on the indexer (128×128, split-K=1)
  and wqkv_a (256×256, split-K=8). All 16 accepted formal measurements had 100
  kernel records. Logs and ordered JSON stage windows confirmed that only AP1
  runs a separate AP0 prebenchmark.

The perf wrapper follows FlyDSL's benchmark order: AP0 runs Torch reference
then formal AP0; AP1 runs Torch reference, an AP0 prebenchmark, then formal AP1.
Each benchmark has its own 2 warmups and 100 timed iterations. The full six-case,
four-repeat performance sweep is a separate user-run measurement:

```bash
AITER_GPU_ARCHS=gfx1250 AITER_JIT_DIR="$(mktemp -d /tmp/aiter_ap0_128x128.XXXXXX)" \
  python -m op_tests.test_mxfp8fp4gemm_perf --apre 0
```

## Use from Python

```python
# A, B and the two scale tensors must already use the existing aiter shuffles.
out = aiter.gemm_a8w8_mxfp8(
    A, B, ScaleA, ScaleB,
    dtype=torch.bfloat16,
    a_preshuffle=True,
    splitk=1,
)
```

For explicit selection, supply:

```python
kernelName="_ZN5aiter48f8gemm_bf16_mxfp8fp8_ABpreShuffle_128x128_4x4_psE"
```

For AP0, pass row-major A, keep the existing B/ScaleA/ScaleB shuffles, and set
`a_preshuffle=False`. Its explicit symbol is:

```python
kernelName="_ZN5aiter47f8gemm_bf16_mxfp8fp8_BpreShuffle_128x128_4x4_psE"
```

## Native op test

Run only after checking that the shared GPU is idle and its temperature has
recovered. Run function validation before timing:

```bash
AITER_GPU_ARCHS=gfx1250 python op_tests/test_mxfp8fp4gemm.py \
  --mode func --intype a8w8 --apre 1 --outtype bf16 \
  --shape 512,8192,1536 --splitk 1 --no-reduce \
  --data-init poc --scale-init poc --seed 0 --warmup 0 --iters 5 --rotate 1

AITER_GPU_ARCHS=gfx1250 python op_tests/test_mxfp8fp4gemm.py \
  --mode perf --intype a8w8 --apre 1 --outtype bf16 \
  --shape 512,8192,1536 --splitk 1 --no-reduce \
  --data-init poc --scale-init poc --seed 0 --warmup 50 --iters 200 --rotate 30
```

Without `--knl-name`, these exercise actual C++ automatic dispatch. Explicit
selection uses `--knl-name` followed by the symbol above. The JSON output
records raw profiler kernel counts; each timed iteration must contain exactly
one intended ASM kernel. `--no-reduce` retains the prior benchmarking option
for validating partial split-K planes in the 256×256 variants.

## Rebuild and registration

Each shipped assembly preserves its POC variant's executable `.text` byte for
byte. Provenance and hashes are in
`hsa/gfx1250/mxfp8fp4gemm/src/128x128_k128_pf8.json` (AP1) and
`hsa/gfx1250/mxfp8fp4gemm/src/128x128_k128_pf8_ap0.json` (AP0).

```bash
bash hsa/gfx1250/mxfp8fp4gemm/src/build_128x128.sh
```

This rebuilds both `.co` files; it does not run a GPU workload or replace the
CSV. Each layout has its own CSV row. Use a fresh
`AITER_JIT_DIR` when testing the changed CSV/C++ so codegen and the HIP module
are rebuilt, for example a new directory under the integration validation
output. Do not reuse the old measurement cache.
