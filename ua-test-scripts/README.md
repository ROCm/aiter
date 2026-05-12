# Unified Attention Testing Suite

Correctness and performance tools for the CK Tile unified attention kernel
(`aiter.ops.unified_attention.unified_attention_fwd`), with a Triton
unified attention reference for apples-to-apples comparison.

## What's in the kernel today

`unified_attention_fwd` is a thin Python wrapper around the C++/HIP
`module_unified_attention` JIT op. It transparently applies a
**FlashDecoding-style KV split** when the heuristic says batch/CTA
occupancy is too low to saturate the GPU:

1. **`_pick_num_splits`** chooses `num_splits ∈ [1, 16]` from a cheap
   pure-CPU CTA-occupancy estimate (no device sync — safe under CUDA
   graph capture).
2. **`num_splits == 1`** → kernel writes directly to `output`.
3. **`num_splits > 1`** → kernel writes per-split partials into FP32
   `o_acc` / `lse_acc` workspaces. A single fused Triton kernel
   (`reduce_segments_ck_layout`) then does the LSE merge into `output`.

The Triton combine is the CK-layout sibling of Triton-UA's own
`reduce_segments`: algorithmically identical math, just adapted to CK's
workspace layout. This means head-to-head CK-vs-Triton benchmarks pay the
**same** combine cost on both sides, so the comparison cleanly isolates
the attention kernel.

The kernel supports:

| head_dim | num_queries_per_kv | dtypes        | mask    |
|---------:|-------------------:|---------------|---------|
| 64       | 1, 2, 4, 8, …      | fp16, bf16    | causal / none |
| 128      | 1, 2, 4, 8, …      | fp16, bf16    | causal / none |

It does **not** support: fp8 quant, sliding window, softcap, attention
sinks. Triton-UA covers those.

## Quick start

### Single-shape correctness check (CK output matches Triton)

```bash
python test_single_shape.py -b 256 -sq 1 -sk 8192 -hq 64 -hk 8 -d 64 --test
```

`--test` cross-checks the CK output against Triton using
`torch.testing.assert_close(atol=0.015, rtol=0.01)`, same tolerance as the
broader CK/Triton test suites for non-quantized dtypes.

### Single-shape benchmark (CK vs Triton)

```bash
python test_single_shape.py -b 256 -sq 1 -sk 8192 -hq 64 -hk 8 -d 64 \
  --warmup 20 --iters 100
```

CUDA-graph timing is on by default (`--no-graph` to disable). Output
reports CK time, Triton time, speedup, and effective HBM bandwidth for
both.

### Full correctness suite

```bash
python -m pytest test_unified_attention_ck_correctness.py -q
```

Expected: **241 passed, 4 failed**. The 4 failures are the same
`num_blocks=32768 + d=128 + block_size=64` int32-stride-overflow cases on
the rebased-pointer path, tracked separately — they are not regressions.

The suite includes both the default (transparent split-KV) path
(`test_ck_unified_attn`) and an explicit-num_splits sweep that exercises
the kernel's per-split write path directly (`test_ck_unified_attn_splitkv`).

### Reproducing the published perf numbers

See [`PERFORMANCE.md`](./PERFORMANCE.md) for the full CK-vs-Triton sweep
at `sk=120k` and the commands to reproduce each row.

## Useful environment variables

| variable                  | effect                                                       |
|---------------------------|--------------------------------------------------------------|
| `AITER_UA_FORCE_SPLITS=N` | Bypasses `_pick_num_splits`; forces `num_splits=N`. `1` disables the combine path entirely. Any integer in `[1, 16]` works (no need for power of 2). |
| `AITER_REBUILD=1`         | Forces a full cold rebuild of `module_unified_attention.so`. |
| `AITER_REBUILD=2`         | Incremental rebuild via Ninja (much faster after a code change). |

## Command-line arguments (`test_single_shape.py`)

### Required shape parameters

- `-b, --batch` — number of sequences
- `-sq, --seqlen-q` — query sequence length (`1` = decode)
- `-sk, --seqlen-k` — KV / context length
- `-hq, --num-q-heads` — number of query heads
- `-hk, --num-kv-heads` — number of KV heads (`hq == hk` = MHA, `hq > hk`
  with `hq % hk == 0` = GQA)
- `-d, --head-size` — head dimension (64 or 128)

### Testing / benchmarking options

- `--test` — run correctness check before benchmarking
- `--warmup N` — warmup iterations (default: 10)
- `--iters N` — benchmark iterations (default: 50)
- `--no-graph` — disable CUDA-graph timing (eager mode)
- `--only-ck` / `--only-triton` — skip the other backend

### Optional shape parameters

- `--block-size N` — KV cache page size (default: 32)
- `--num-blocks N` — KV cache pool size (default: auto)
- `--dtype {bf16,fp16}` — data type (default: bf16)
- `--window-left N` — sliding window left (default: -1 = no window)
- `--softcap F` — softcap value (default: 0.0 = disabled)

## Files

| file                                          | purpose                                   |
|-----------------------------------------------|-------------------------------------------|
| `test_single_shape.py`                        | Single-shape correctness + benchmark      |
| `test_unified_attention_ck_correctness.py`    | pytest correctness sweep (default + split-KV) |
| `test_split_kv.py`                            | Standalone split-KV correctness/perf probe |
| `rocprof_bench.sh`                            | ROCProfiler v3 wrapper                    |
| `parse_kernel_trace.py`                       | Kernel-trace parser (CK + Triton 2D/3D)   |
| `parse_rocprof.py`                            | Alternative stats parser                  |
| `bench_ck_vs_triton_csv_rows.py`              | Batch testing from a CSV of shapes        |
| `pawel-2d-3d_50rows_verified.csv`             | Reference shape sweep                     |
| `PERFORMANCE.md`                              | Headline perf numbers + reproduction commands |
| `README.md`                                   | (this file)                               |

## Where the code lives

- **Python entry point**:
  `aiter/ops/unified_attention.py` (`unified_attention_fwd`,
  `_pick_num_splits`, `_combine_splits`)
- **JIT-compiled C++/HIP kernel**: `module_unified_attention` (built from
  `csrc/py_itfs_ck/unified_attention_ck_kernels.cu` →
  `3rdparty/composable_kernel/example/ck_tile/42_unified_attention/…`)
- **Triton combine for the CK split path**:
  `aiter/ops/triton/_triton_kernels/attention/unified_attention.py`
  (`reduce_segments_ck_layout`, right next to Triton-UA's own
  `reduce_segments`)
- **Triton reference**:
  `aiter/ops/triton/attention/unified_attention.py`

## Hardware

Numbers in `PERFORMANCE.md` are from AMD MI350 (gfx950, 304 CUs) with
ROCm 7.0.0. The kernel and the heuristic are device-agnostic — the
heuristic reads `multi_processor_count` at call time — but absolute
timings will obviously differ on other hardware.
