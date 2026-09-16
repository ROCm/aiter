# gfx1250 MXFP8 GEMM 128×128 K128/PF8

The `(M,N,K)=(512,8192,1536)` FP8×FP8 case with `a_preshuffle=True`
now selects the 128×128 K128/PF8 kernel. Other shapes retain the existing
tile preferences. If the 128×128 CSV row is absent, this shape falls back to
256×256. Output is BF16.

## Kernel configuration

| Setting | Value |
|---|---|
| Output tile / K step / prefetch stages | 128×128 / 128 / 8 |
| A / B preshuffle | 1 / 1 |
| Cluster X×Y / traversal grid X×Y | 4×4 / 8×2 |
| Persistent workgroups / threads per workgroup | 256 / 128 |
| Input early timeout / output descriptor ET | 1 / 1 |
| A and ScaleA | HT/CU |
| B and ScaleB | RT_NT/CU |
| D | BYPASS/SYS |
| LDS / VGPR allocation | 278528 bytes / 896 |
| Packed kernargs / preload | 80 bytes / 20 dwords |
| Split-K capability | None; `splitk=1` only |

The CSV `splitk=0` is a capability flag, not the runtime split count.
The public Python API resolves automatic split-K to 1 for this variant and
returns the GEMM output directly without `torch.sum`.

Explicit selection currently requires positive M/N multiples of 128 and
K≥1024, K%128=0. Eight K stages are preloaded; shorter K and partial output
tiles have not been enabled. FP4 and A without preshuffle keep their existing
variants. No generalized performance preference is inferred from this one
measured shape.

## Validation and performance

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

The shipped assembly is generated from the selected POC preset; only its
entry symbol has changed. The executable `.text` matches the validated POC
code object byte for byte. Provenance and hashes are in
`hsa/gfx1250/mxfp8fp4gemm/src/128x128_k128_pf8.json`.

```bash
bash hsa/gfx1250/mxfp8fp4gemm/src/build_128x128.sh
```

This rebuilds only the `.co`; it does not run a GPU workload or replace the
CSV. The one new CSV row is committed with the dispatch change. Use a fresh
`AITER_JIT_DIR` when testing the changed CSV/C++ so codegen and the HIP module
are rebuilt, for example a new directory under the integration validation
output. Do not reuse the old measurement cache.
