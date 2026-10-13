# Kimi-K3 merged-front tuning for hipBLASLt 1.4.1

`gfx950_kimik3_bf16_fp32.csv` contains 58 exact-M selections for the merged
MoE front in [vLLM #55811](https://github.com/vllm-project/vllm/pull/55811).
It uses BF16 inputs, FP32 output, N=6016, K=7168, no bias, no scaling,
and no weight preshuffle, on MI355X (gfx950, 256 CUs).

The table is specific to **hipBLASLt 1.4.1, revision 8d1ae90e**. Solution
indices are not portable across hipBLASLt builds. The older ROCm 7.2.3
build (1.2.2.70203-90~22.04, revision dabb6df2b9) uses different indices;
its bundled selections produced `HIPBLAS_STATUS_INVALID_VALUE` with
this ROCm 10 runtime. AITER's version alone does not identify the BLAS build.

This directory is outside `model_configs/`, so these rows are not merged
automatically into the default table. Select them explicitly only with
the matching runtime. The table contains only the retuned shapes; other
shapes should retain the deployment's existing tuning.

## Apply the selections

Obtain the resolved BF16 tuning CSV from the intended deployment and use
it as `BASE_CSV`. Apply this table by exact key, replacing the old rows
regardless of their recorded latency. Latencies from different builds
and measurement protocols cannot determine which version's row to keep.
Do not concatenate the files or pass them as a colon-separated config:
that creates duplicate keys and can select an incompatible old index.

```bash
BASE_CSV=/absolute/path/to/resolved-bf16-tuned-gemm.csv
OVERLAY_CSV=/absolute/path/to/gfx950_kimik3_bf16_fp32.csv
OUTPUT_CSV=/absolute/path/to/kimik3-hipblaslt-1.4.1-8d1ae90e.csv
python3 - "$BASE_CSV" "$OVERLAY_CSV" "$OUTPUT_CSV" <<'PY'
import csv
import sys
from pathlib import Path

base, overlay, output = map(Path, sys.argv[1:])
assert output.resolve() not in (base.resolve(), overlay.resolve())
with base.open(newline="") as stream:
    reader = csv.DictReader(stream)
    fields = reader.fieldnames
    rows = list(reader)
with overlay.open(newline="") as stream:
    reader = csv.DictReader(stream)
    assert reader.fieldnames == fields
    updates = list(reader)
keys = fields[:10]  # gfx through bpreshuffle, the AITER lookup key
def key(row):
    return tuple(row[name] for name in keys)
assert len({key(row) for row in updates}) == len(updates)
replacement = {key(row): row for row in updates}
merged = [replacement.pop(key(row), row) for row in rows]
merged.extend(replacement.values())
assert len({key(row) for row in merged}) == len(merged)
with output.open("x", newline="") as stream:
    writer = csv.DictWriter(stream, fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(merged)
PY
export AITER_CONFIG_GEMM_BF16="$OUTPUT_CSV"
```

Set the variable before starting Python; AITER caches tuning lookups.
The example preserves existing rows outside the 58 exact keys and refuses
to overwrite the input files or an existing output.

## Tuning and validation

Measured with AITER v0.1.24.post1, PyTorch 2.12.0+rocm10.0.0 and the
October 12, 2026 vLLM nightly-rocm100 image:

```text
image digest: sha256:94e39a6e8237e21877efb70c8a33485557c051eac892cdee425483a7f5c0257f
libhipblaslt.so.1 SHA256: 726921b73776933606610d0d812fcbe4d448423b16057352f7c9902b152cda38
vLLM commit: fe4df3df0eb703862c686b256e9fe44fa94c8945
vLLM PR base: d1447951e396ea267216e2386da26fadf0ab44b3
```

The original 28 selections are preserved. Thirty additional M values were
retuned on the same library build:

```text
decode: 18, 20, 24, 25, 30, 35, 36, 40, 42, 45,
        50, 55, 60, 65, 70, 75, 85, 90, 95, 100
prefill: 256, 384, 768, 1280, 1792, 2560, 3072, 4096, 6144, 8192
```

The decode additions cover complete DSpark verification batches for C4
(k=5, up to 8 active requests) and C10 (k=4, up to 20). The prefill values
are representative candidates, not an exhaustive list of runtime token
counts or a measured workload hit histogram. The matching vLLM whitelist
must also admit a token count for its merged-front row to be used.

For each added M, all 684 enumerated hipBLASLt candidates plus
`torch.mm(out_dtype=torch.float32)` were checked and measured: 20,550 new
candidate/shape pairs with zero rejected candidates. Two full passes used
independently shuffled order, three samples per pass, and 32 GEMMs per
captured graph. Each graph rotated through eight separate input/weight/output
buffers (689,963,008 bytes of weights) to represent different model layers.
Candidates for each M stayed on one physical GPU. The original 28 rows used
the same protocol, giving 39,730 candidate/shape pairs across both runs.

The top 16, every candidate within 3% of the scan winner, the prior lookup
choice, and torch.mm received nine interleaved confirmation measurements
with fresh random inputs. The lowest confirmation median supplies `us`.
The selected kernels passed on all eight rotating buffers. A fresh process
verified all 58 exact keys and actual vLLM merged-GEMM dispatch on two held-out
seeds: 116 checks, maximum absolute error 0.001953125.

M=512, M=1024 and M=4096 select torch.mm because it won the comparison.
The other 55 rows select hipBLASLt. These are measured winners for this
build and memory-access protocol; near ties can change with measurement noise.

The extension ran in Slurm job 212963 on `crsuse2-m2m-v3-037`, with eight
exclusive MI355X GPUs. All 4,148 rows of the prior deployment CSV remain
byte-identical; the extension appends 30 rows, giving 4,178 total rows.

```text
58-row table SHA256: 39951379f24ed68cc9f4a4933dbd9f511eac309d4d8e206b2d6305430f1fa7b3
full deployment CSV SHA256: b7954c5c9fcc12645946d0bd8c2b1b8184decacef9a769ea6c40cbe54cc5d582
```

## Front comparison

The actual vLLM merged front (GEMM plus SiTU/split) was compared with the
native three projections and SiTU using the same eight rotating weight sets,
32 front calls per graph, and nine interleaved measurements. All 58 shapes
passed 464 buffer checks, 58 changed-input CUDA Graph replay checks, and
workspace alias/boundary checks. GEMMs were checked against FP32 references;
the SiTU/split checks use the companion AITER test tolerances. Native BF16
projections can round differently before SiTU; maximum relative L2 difference
over the native comparisons was 0.007148. The existing vLLM front suite
passed 73 tests (20 unrelated tests deselected).

| New M | Merged front (us) | Native front (us) | Native / merged |
|---|---:|---:|---:|
| 18 | 23.065 | 36.360 | 1.576x |
| 20 | 23.215 | 36.467 | 1.571x |
| 24 | 23.586 | 34.473 | 1.462x |
| 25 | 23.588 | 34.402 | 1.458x |
| 30 | 23.780 | 35.274 | 1.483x |
| 35 | 25.619 | 35.917 | 1.402x |
| 36 | 25.535 | 36.035 | 1.411x |
| 40 | 26.079 | 36.705 | 1.407x |
| 42 | 26.290 | 36.928 | 1.405x |
| 45 | 26.169 | 37.344 | 1.427x |
| 50 | 26.506 | 38.065 | 1.436x |
| 55 | 26.308 | 38.647 | 1.469x |
| 60 | 26.900 | 39.509 | 1.469x |
| 65 | 28.936 | 43.434 | 1.501x |
| 70 | 28.851 | 43.698 | 1.515x |
| 75 | 29.063 | 44.227 | 1.522x |
| 85 | 29.193 | 45.149 | 1.547x |
| 90 | 29.485 | 45.838 | 1.555x |
| 95 | 29.610 | 46.432 | 1.568x |
| 100 | 31.969 | 49.294 | 1.542x |
| 256 | 43.390 | 71.342 | 1.644x |
| 384 | 51.244 | 101.666 | 1.984x |
| 768 | 80.613 | 128.354 | 1.592x |
| 1280 | 107.323 | 179.911 | 1.676x |
| 1792 | 139.059 | 195.640 | 1.407x |
| 2560 | 188.758 | 255.432 | 1.353x |
| 3072 | 235.703 | 273.216 | 1.159x |
| 4096 | 298.219 | 318.733 | 1.069x |
| 6144 | 458.951 | 465.501 | 1.014x |
| 8192 | 588.782 | 548.256 | 0.931x |

**M=8192 keeps its measured CSV entry but remains outside the vLLM merged
whitelist:** the full front was 7.39% slower than native. The other 29 new
shapes enter the companion whitelist, giving 57 enabled shapes and max M=6144.
The shared front workspace is then 213 MiB per worker (up from 71 MiB);
all layers share it. M=6144 has only a 1.43% speedup in this microbenchmark.
These measurements do not establish AgenticX end-to-end gains.

## Historical end-to-end check (28 shapes)

The results below used the original 28-row table at commit
`089f8e0b2a4fda91fb13d9b9e2675daf005dcee4`, with full deployment CSV SHA256
`7d8529910adfa46cf02e1640d867d359bfb558feb403426f54fb873a73052243`.
They do not measure the expanded CSV and whitelist.

Both arms used the same updated table, on one 8-GPU MI355X node, TP8/DCP1,
BF16 linear projections, FP8 KV cache and DSpark (7 speculative tokens,
synthetic acceptance length 3.84). Prefix caching was disabled. Each arm
had three full-length warmup requests and 20 measured requests of exactly
8192 input / 1024 output tokens, concurrency 1. Both passed 20/20 requests
with no failures; exact request intervals confirmed no overlap. The
merged-front path executed during FULL CUDA Graph capture in the PR arm.

| Metric | Nightly + updated CSV | Nightly + vLLM PR + updated CSV | Change |
|---|---:|---:|---:|
| Output tokens/s | 175.065 | 183.244 | +4.67% |
| Mean TTFT (ms) | 439.050 | 435.498 | -0.81% |
| Mean TPOT (ms) | 5.288 | 5.037 | -4.76% |
| P90 TPOT (ms) | 5.389 | 5.146 | -4.51% |

This compares the vLLM PR against nightly while holding the new CSV fixed;
it does not isolate the CSV change's end-to-end contribution. The run is a
20-request fixed-length check, not a 3600-second AgenticX result.
