# Kimi-K3 merged-front tuning for hipBLASLt 1.4.1

`gfx950_kimik3_bf16_fp32.csv` contains 28 exact-M selections for the merged
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
The example preserves existing rows outside the 28 exact keys and refuses
to overwrite the input files or an existing output.

## Tuning and validation

Measured with AITER v0.1.24.post1, PyTorch 2.12.0+rocm10.0.0 and the
October 12, 2026 vLLM nightly-rocm100 image:

```text
image digest: sha256:94e39a6e8237e21877efb70c8a33485557c051eac892cdee425483a7f5c0257f
libhipblaslt.so.1 SHA256: 726921b73776933606610d0d812fcbe4d448423b16057352f7c9902b152cda38
vLLM commit: fe4df3df0eb703862c686b256e9fe44fa94c8945
vLLM PR head: d1447951e396ea267216e2386da26fadf0ab44b3
```

For each M, all 684 enumerated hipBLASLt candidates plus
`torch.mm(out_dtype=torch.float32)` were checked and measured. Two full
passes used independently shuffled order, three samples per pass, and
32 GEMMs per captured graph. Each graph rotated through eight separate
input/weight/output buffers (689,963,008 bytes of weights) to represent
different model layers. Candidates for each M stayed on one physical GPU.

The top 16, every candidate within 3% of the scan winner, the previous
choice, and torch.mm received nine interleaved confirmation measurements
with fresh random inputs. The lowest confirmation median supplies `us`.
There were 19,180 candidate/shape pairs and zero rejected candidates.
The selected kernels passed on all eight buffers, followed by 56 fresh
process exact-lookup and actual PR-dispatch checks with two held-out seeds
per shape (maximum absolute error 0.001525879).

M=512 and M=1024 select torch.mm because it won the comparison. The other
26 rows select hipBLASLt. These are measured winners for this build and
memory-access protocol; near ties can change with measurement noise.

## End-to-end check

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
