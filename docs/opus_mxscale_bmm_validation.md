# MXScale BMM review validation (gfx950, 2026-09-24)

The machine-readable observations, binary hashes, and scoped timing samples
are in [opus_mxscale_bmm_evidence.json](opus_mxscale_bmm_evidence.json).
See [scale groups and compatibility](opus_mxscale_bmm.md) for the public API
and the limits of the performance evidence.

## Correctness

| Check | Result |
| --- | --- |
| Interface and config-shape collision tests | 55 passed |
| GPU regressions | 46 passed |
| Installed-module raw error/public-entry smoke tests | 9 passed |
| Unknown kid on a fresh thread, prior module | SIGABRT reproduced in a contained subprocess |
| Unknown kid on a fresh thread, corrected module | RuntimeError caught; process survives |
| Host-only module rebuild | All 253 GPU code objects have identical `.text` hashes |
| Model tuned CSV | All 167 rows retained; no retuning or source rewrite in this review fix |

GPU coverage includes GS32/GS128 public calls, the default GS128 argument,
direct and allocated split-K workspace paths, eager and `torch.compile`
(`backend="aot_eager", fullgraph=True`), scale-ring determinism, invalid
workspace rejection, and large/partial batch-pair panels. This is compiler
boundary coverage; it is not an Inductor performance measurement.

## Large-grid barrier check

Kids 8646 and 9646 were run at B=4, M=16384, N=1024, K=4096, split-K=1.
Each kid/output-dtype combination used seeds 77 and 991 and ten repeated
launches per seed. Every output was finite, all repeated results were bitwise
equal, and no subprocess reached its 180-second timeout.

| Kid | Group | Output | Maximum mismatch fraction across seeds |
| ---: | ---: | --- | ---: |
| 8646 | 128 | BF16 | 0.000317528844 |
| 9646 | 32 | BF16 | 0.000317454338 |
| 8646 | 128 | FP32 | 0.000297710299 |
| 9646 | 32 | FP32 | 0.000297650695 |

Mismatch is the fraction failing `torch.isclose(rtol=0.01, atol=0.01)`
against the dequantized FP32 reference, with an acceptance limit of 0.001.
This finite sweep provides regression evidence, not proof for every possible
GPU scheduling interleaving.

## Reproduction

Run from the repository root with the OPUS module built for gfx950 and choose
an available GPU. `AITER_REBUILD=0` below reuses that build.

```bash
HIP_VISIBLE_DEVICES=0 AITER_REBUILD=0 python3 -m pytest -q \
  op_tests/test_opus_a8w8_interface.py \
  op_tests/tuning_tests/test_config_shape_collision.py

HIP_VISIBLE_DEVICES=0 AITER_REBUILD=0 python3 -m pytest -q -s \
  op_tests/test_opus_mxscale_regressions.py
```

The barrier test emits one JSON record per kid/dtype/seed and contains each
case in a bounded subprocess. The raw unknown-kid test also uses a subprocess
and disables core dumps so an old binary fails the test without terminating
the pytest process.

**Exact PR-base/head GS128 timing and downstream serving performance remain
unverified.** Historical model-CSV timings are not a no-regression result.
The attached six-round timing samples cover only the earlier local batch-pair
traversal experiment and identify the compared binaries by SHA256.
