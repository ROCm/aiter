# A0 native CO launch test on g21-10

## Exact environment

```text
Image: rocm/dev-ubuntu-24.04:10.1.0-full
Image digest: sha256:5ed1362ea542a928651e4c710b44024ebe98870f74159cb70f0265aec8ef0abe
HIP runtime / SDK: 7.16.26385
SDK compiler: AMD clang 24.0.0git @ d6f6cb691863e9258034ab48c1dfce21689d5606
Machine: g21-10
Container: blyu_a0_devfull_g21_10_20261027
HIP_VISIBLE_DEVICES: 3 (test process ordinal 0)
Reported device: gfx1250, ASIC Revision 0
Test directory: /home/blyu/opus-a0-devimage-20261027
```

The device code objects are the already-built `amdgpu-pin-hybrid` A0 files, not
new binaries compiled with this SDK. Their actual ELF processor is `0xeb` and
the current host loader automatically selected `co/gfx1250-a0/` from ASIC
Revision 0. No processor flag was rewritten.

The test is native C++/HIP. It does not import PyTorch, call rocminfo, use GPU RNG,
or depend on a GPU reference GEMM. CPU code prepares exact BF16 inputs and checks
all returned elements against an FP32 CPU golden (`rtol=0.03`, `atol=0.5`). Each
case starts with a NaN-poisoned output and launches twice to check reinitialization.

## Completed smoke result

The initial smoke run completed successfully:

| kid | family | completed shapes (M/N/K) | batch | row padding | max absolute error |
|---:|---|---|---:|---:|---:|
| 21016 | compute, c1x1 | 1/1/1; 129/129/385 | 2 | 0; 8 | 0 |
| 21105 | wave-layout, c1x1 | 1/1/1; 129/129/385 | 2 | 0; 8 | 0 |
| 21316 | ring/pin, c4x4 | 1/1/1; 257/257/257 | 2 | 0; 8 | 0 |

```text
A0_NATIVE_CO_LAUNCH_PASS variants=3 cases=6
```

This proves actual strict-image loading, native HIP launch, stream completion,
output copying and numerical agreement in the specified image for these cases.
It is not a claim about performance or every registered variant.

## Full-run limitation

A subsequent attempt over all 221 variants did not complete. It passed the first
five cases, then stopped producing progress after `LOAD_OK kid=21316` for the
257/257/257, batch=2, pad=8 case—the same case that passed in the initial smoke.
The process exceeded its 420-second test window. The launch test has since been
instrumented with separate launch-submitted, stream-complete and output-copy
markers to distinguish GPU waits from CPU golden work on a future reproduction.

There is no complete all-221 native correctness certification from this run.
This repeat-run behavior requires further investigation under an available,
healthy device window; it must not be presented as all kernels passing. Only
this test's own process was targeted for cancellation, with no shared GPU reset
or other-user workload interruption. The native worker did not exit after
termination requests: container PID 90 / host PID 1851629 remained active. It
belongs to the private container above; no additional test was started while it
remained present. Its device/driver wait needs to be resolved before a new GPU
validation run.

## Reproduce

The cases file is generated from the actual A0 build manifest and contains all
221 kids with unchanged names, tiles, slots and cluster sizes. The first three
rows select the family representatives above.

```sh
# Inside the exact image, with the private source tree and CO root mounted:
/opt/rocm/bin/hipcc -x hip --offload-arch=gfx1250 -std=c++20 \
  -DENABLE_CK=0 -DAITER_ENABLE_CLUSTER_LAUNCH \
  -I/work/source/csrc/include -I/work/source/csrc/opus_gemm/include \
  /work/test_opus_a0_co_launch.cpp -o /work/test_opus_a0_co_launch

# Device kernels remain the strict 0xeb CO files. This command only builds
# the native host tester; no B0 device kernel is launched by that target flag.
timeout -k 10s 120s /work/test_opus_a0_co_launch /work/co /work/cases.tsv 0 3

# Full variant list, only after the repeat-run wait has been investigated:
timeout -k 10s 420s /work/test_opus_a0_co_launch /work/co /work/cases.tsv 0 0
```

Native tester source: `op_tests/cpp/test_opus_a0_co_launch.cpp`.
Raw first-run evidence is retained in the task directory as `smoke.log`; the
incomplete subsequent run is `all-kernels.log`. Image configuration and host
compile diagnostics are also retained.
