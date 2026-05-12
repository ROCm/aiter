# CK Unified Attention — Performance vs Triton

Performance comparison of the CK Tile unified attention kernel against the
Triton unified attention reference, after the recent refactor work
(transparent split-KV, runtime `kBlockQ`, collapsed MHA/GQA variants,
m32/m16 d=128 decode variants, Triton-fused combine).

All numbers measured on **AMD MI350 (gfx950)**, 304 CUs, ROCm 7.0.0, bf16,
context length `sk=120000`, decode (`sq=1`), `block_size=16`, CUDA-graph
timing. Each cell is `time_triton / time_ck` (i.e. `>1.0` = CK wins).

## Headline numbers

### d=64 GQA-8 (`hq=64`, `hk=8`)

| batch | CK time (ms) | Triton time (ms) | Speedup    | CK bandwidth |
|------:|-------------:|-----------------:|-----------:|-------------:|
|     4 | 0.276        | 0.498            | **1.82x**  | 3.56 TB/s    |
|     8 | 0.414        | 1.015            | **2.45x**  | 4.75 TB/s    |
|    32 | 1.577        | 3.010            | **1.91x**  | 4.99 TB/s    |
|    64 | 3.134        | 5.485            | **1.75x**  | 5.02 TB/s    |
|   128 | 6.236        | 10.335           | **1.66x**  | 5.05 TB/s    |
|   256 | 12.444       | 15.935           | **1.28x**  | 5.06 TB/s    |

### d=128 MHA (`hq=16`, `hk=16`)

| batch | CK time (ms) | Triton time (ms) | Speedup    | CK bandwidth |
|------:|-------------:|-----------------:|-----------:|-------------:|
|     4 | 0.767        | 0.815            | **1.06x**  | 5.13 TB/s    |
|     8 | 1.508        | 2.023            | **1.34x**  | 5.21 TB/s    |
|    32 | 5.895        | 6.821            | **1.16x**  | 5.34 TB/s    |
|    64 | 11.710       | 10.892           |   0.93x    | 5.37 TB/s    |
|   128 | 23.278       | 25.401           | **1.09x**  | 5.41 TB/s    |
|   256 | 46.507       | 47.170           | **1.01x**  | 5.41 TB/s    |

CK saturates ~5 TB/s effective HBM bandwidth across the sweep (MI350 peak
is ~5.3 TB/s). The two regimes where Triton remains competitive — d=64
b=256 and d=128 b=64 — are also the two cases where Triton's 2D path
already saturates the device, leaving no headroom for split-KV to help.

## Why these numbers (architectural recap)

The CK kernel runs under a **transparent split-KV** wrapper
(`unified_attention_fwd` in `aiter/ops/unified_attention.py`):

1. **`_pick_num_splits`** picks `num_splits ∈ [1, 16]` from a CTA-occupancy
   heuristic — purely CPU-side, no device sync, so it's CUDA-graph safe.
   Target is roughly `2 × num_CUs` total CTAs.
2. If `num_splits == 1` the kernel writes directly to the output tensor
   (single-launch path).
3. If `num_splits > 1` the kernel writes per-split partials to FP32
   workspaces (`o_acc`, `lse_acc`), then **`_combine_splits`** does a
   FlashDecoding-style LSE merge.

The combine is a single fused Triton kernel (`reduce_segments_ck_layout`),
algorithmically identical to Triton-UA's own `reduce_segments`:

```
lse_max  = max_s lse[s]
w[s]     = exp(lse[s] - lse_max)    # -inf rows -> 0
out      = Σ_s o_acc[s] · w[s] / Σ_s w[s]
```

The only differences vs `reduce_segments` are layout (head-major
`[H, S, T, D]` instead of token-major `[T, H, S, D]`) and lse encoding
(single natural-log `lse_acc` instead of separate base-2 `m`/`expsum`).
Because both backends now use the same kernel for the combine step, the
table above isolates the **attention-kernel** comparison from any combine
overhead difference.

## Reproducing the numbers

```bash
# d=64 GQA-8
for B in 4 8 32 64 128 256; do
  python ua-test-scripts/test_single_shape.py \
    -b $B -sq 1 -sk 120000 -hq 64 -hk 8 -d 64 \
    --block-size 16 --num-blocks 120000 \
    --test --warmup 20 --iters 100
done

# d=128 MHA
for B in 4 8 32 64 128 256; do
  python ua-test-scripts/test_single_shape.py \
    -b $B -sq 1 -sk 120000 -hq 16 -hk 16 -d 128 \
    --block-size 16 --num-blocks 60000 \
    --test --warmup 20 --iters 100
done
```

`--test` runs correctness vs Triton in-process before benchmarking, so each
data point in the sweep is guaranteed to match Triton's output within
`atol=0.015, rtol=0.01` (same tolerance as the broader Triton/CK test
suites for non-quantized dtypes).

> The d=128 sweep uses `--num-blocks 60000` (not 120000) to stay below the
> int32 stride-overflow threshold in the kernel's rebased-pointer path —
> this is a known kernel limitation, see the 4 expected failures in
> `test_unified_attention_ck_correctness.py`.

## Correctness reference

```bash
python -m pytest ua-test-scripts/test_unified_attention_ck_correctness.py -q
```

Expected: **241 passed, 4 failed**. The 4 failures are all
`num_blocks=32768 + d=128 MHA + block_size=64`, caused by int32 overflow in
the rebased-pointer path. Unrelated to split-KV / combine and tracked
separately.

For the split-KV path specifically:

```bash
python -m pytest ua-test-scripts/test_unified_attention_ck_correctness.py::test_ck_unified_attn_splitkv -q
```

## Tuning levers (advanced)

The split-KV heuristic can be overridden via environment variable for
A/B testing or profiling:

```bash
# Force a specific num_splits (1 = single-launch, no combine).
AITER_UA_FORCE_SPLITS=1  python ua-test-scripts/test_single_shape.py ...
AITER_UA_FORCE_SPLITS=8  python ua-test-scripts/test_single_shape.py ...
AITER_UA_FORCE_SPLITS=16 python ua-test-scripts/test_single_shape.py ...
```

Or from Python, callers can pass `allow_splitkv=False` to
`unified_attention_fwd` to disable the transparent split-KV path entirely,
or pass explicit `num_splits` + workspaces to own the combine themselves.

## Hardware

Tested on AMD MI350 (gfx950) with ROCm 7.0.0. The kernel and the heuristic
are not MI350-specific — `_pick_num_splits` reads `multi_processor_count`
from the device at call time — but the absolute timings above are MI350
data points.
