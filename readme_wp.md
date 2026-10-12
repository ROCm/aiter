# WP Gluon GEMM on gfx1250

This branch packages the CDNA5 Gluon warp-pipeline (WP) MXFP8 GEMM in AITER.
It computes `Y = scaled(A) @ scaled(W).T`, with FP8 E4M3 inputs, native K128
E8M0 scales, and BF16 output. It requires a gfx1250 GPU, ROCm PyTorch, and an
upstream Triton build with CDNA5 warp-pipeline/TDM support.

The development environment is Docker container `aiter-triton-main-compare`,
repository `/app/aiter`, branch `comparison/upstream-warp-pipeline`.
The tail change described below was committed as `fe29989af`. The previously
published image `rocm/fw-bringup-mlperf:gfx1250-atom-20261004-gluonwp-20261009`
predates that commit; a new image snapshot is needed to share this exact revision.
The tested Triton version is 3.9.0, from upstream commit
`8262c9a91a1d6828ad4f36437fa0046daa67720d`.

## Source locations

All paths below are relative to the AITER repository root (`/app/aiter` in Docker):

- [Kernel source](aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/gemm_warp_pipeline_cdna5.py):
  BK256, BK512, and BK1024 kernels, flat variants, scale loads, and output stores.
- [Public API](aiter/ops/triton/gemm/basic/gemm_warp_pipeline_cdna5.py):
  `prepare_gemm_warp_pipeline` and `gemm_warp_pipeline_cdna5`.
- [Preparation and launch adapter](aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/warp_pipeline/adapter.py):
  input packing, scale layout, descriptors, and kernel launch.
- [Selected configs](aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/warp_pipeline/configs.json):
  per-shape defaults, selected automatically by the public preparation API.
- [Benchmark and verification CLI](op_tests/op_benchmarks/triton/bench_gemm_warp_pipeline.py).
- [Stock input generator](op_tests/op_benchmarks/triton/bench_gemm_afp8wfp8.py): `generate_inputs`.

The packaged kernel is loaded from AITER, rather than from a separate upstream
example checkout. Preparation happens before timing; a prepared object can be
called repeatedly to launch GEMM into its preallocated output.

## Selected configurations

These are the current best-known **native K128, M-contiguous** defaults from the
shape tuning, not a guarantee of the fastest configuration on every machine.
`Tile` is the per-CTA M x N x K tile. `Cluster` gives the CTA arrangement, so
4x4 means 16 CTAs and 2x2 means 4 CTAs. Internal kernel `BLOCK_M` and `BLOCK_N`
cover the cluster and are larger than the per-CTA tile.

| M | N | K | Tile (per CTA) | Cluster | Flat main pipeline |
|---:|---:|---:|:---|:---|:---|
| 512 | 6144 | 7168 | 128x128x512 | 4x4 | No |
| 512 | 7168 | 3072 | 128x128x512 | 4x4 | Yes |
| 512 | 8192 | 1536 | 128x128x512 | 4x4 | Yes |
| 512 | 2048 | 7168 | 64x64x1024 | 4x4 | No |
| 512 | 65536 | 1536 | 256x256x256 | 2x2 | No |
| 512 | 7168 | 16384 | 128x128x512 | 4x4 | No |
| 16384 | 6144 | 7168 | 256x256x256 | 4x4 | No |
| 16384 | 7168 | 3072 | 256x256x256 | 4x4 | No |
| 16384 | 8192 | 1536 | 256x256x256 | 4x4 | Yes |
| 16384 | 2048 | 7168 | 256x256x256 | 4x4 | No |
| 16384 | 65536 | 1536 | 256x256x256 | 4x4 | Yes |
| 16384 | 7168 | 16384 | 256x256x256 | 4x4 | No |

All configurations use 8 warps per CTA, `waves_per_eu=2`, native K128 scales,
`scale_mode="mcontiguous"`, and two padded strip output buffers
(`register_split_output=True`, `output_padding=True`). Alternative scale layouts
and output-buffer counts are not supported by the public wrapper.

For the **non-flat BK512 kernel**, the main loop retains its six-tile unroll;
the final two K tiles are flattened into eight K128 load/compute stage pairs.
Conversion to BF16 is inside the last compute stage, the final tail
`cluster_sync()` is removed, and output stores plus their completion wait are
outside the warp pipeline. This applies to the first and sixth table rows.
It does not set their `flat_pipeline` option to true. The compiler may still
reschedule conversion instructions among output preparation and LDS stores.
Other kernel variants retain their own existing tail implementations.

## Inputs and scale layout

The CLI uses the same stock generator and seed as:

```text
python /app/aiter/op_tests/op_benchmarks/triton/bench_gemm_afp8wfp8.py \
  --shape M N K --preshuffle --backend gluon \
  --x_scale_group_size 128 --cudagraph
```

That older command selects the previous Gluon kernel, not this WP kernel.
Its input contents are the reference for these tests; its timing protocol is
different. In particular, its `do_bench_cudagraph(rep=100)` setting is a timing
budget, not a request for 100 distinct rotating buffers.

WP inputs are generated as follows:

- Seed: `torch.manual_seed(0)` by default; override with `--seed`.
- A: `[M,K]`, `torch.randn` (standard normal) cast to `torch.float8_e4m3fn`.
- W: `[N,K]`, generated the same way. GEMM uses the transpose of W.
- A scales: `[M,K/128]` uint8 E8M0 codes drawn uniformly from 125 through 129.
  Each row's 128 consecutive K elements share one scale.
- W scales: `[N/128,K/128]` uint8 E8M0 codes from 125 through 129.
  Each 128x128 W block shares one scale.
- A scale value is `2**(code-127)`, giving 0.25, 0.5, 1, 2, or 4.
- Output: `[M,N]` BF16.

Thus **"norm" describes the FP8 tensor contents**, not the scale distribution.
There is no `--input norm` argument on this CLI: normal input is built in.

The preparation API accepts canonical, unshuffled A/W and row-major uint8
scale tensors. It packs the operands for the selected kernel. Prepared A scales
are M-contiguous (physical `[K/128,M]` order), with no expansion to K32 scales.
W scales retain one value per 128x128 block. The prepared scale element counts
remain equal to the original scale counts. For compatibility with the old
Gluon/FlyDSL A-scale storage, the API also accepts
`a_scales_transposed=True`: the tensor still has logical shape `[M,K/128]`,
but its contiguous storage contains the transposed scale order. The scale
inputs to this API must be uint8 codes.

## Benchmark protocol

Defaults are `--repeats 100 --warmup 20 --seed 0`:

1. Generate the canonical input contents and prepare a separate warmup buffer set.
2. Launch once for compilation and synchronize. This launch is not timed.
3. Run 20 additional warmup GEMMs reusing that one warmup set.
4. Clone the canonical contents and prepare **100 distinct buffer sets**.
   Each set has distinct A, W, A-scale, W-scale, and output storage: 500 distinct
   pointers total. Contents are identical across sets; they are not resampled.
5. Capture 100 GEMMs in one CUDA graph, one launch per set, then replay it once
   for timing with GPU events. Each set is used once in the timed replay.
6. Report `us = total_elapsed_ms * 1000 / 100` and
   `tflops = 2*M*N*K / (us*1e6)`.

Allocation, packing/preshuffling, compilation, warmup, graph construction, and
verification are outside the timed interval. Rotation is always enabled in this
CLI; there is no separate `--rotate-buffer` switch. `--repeats` controls both
buffer-set count and timed GEMM count, not the number of graph replays.
There are no repeated timed rounds over the same 100 sets. Run another process
for another sample; this allocates new sets, but does not guarantee a cold cache.

Use `gpu-lock` for **every GPU execution, including Docker commands and
verification**. Run each shape as a separate locked process so the lock is
released between shapes. Large shapes need enough GPU memory for all 100 sets;
reducing `--repeats` changes the protocol and is not directly comparable.

## Run one shape in Docker

Run these commands on the host. The container must be running and have AITER,
Triton, and GPU access configured. Change `WP_CONTAINER` to your container name.
`gpu-lock` runs on the host and protects the entire Docker invocation.

```bash
WP_CONTAINER=aiter-triton-main-compare

gpu-lock docker exec -w /app/aiter "$WP_CONTAINER" \
  python3 -m op_tests.op_benchmarks.triton.bench_gemm_warp_pipeline \
  --shape 512 6144 7168 --warmup 20 --repeats 100 --seed 0
```

Replace the three shape values with any row from the configuration table. The
CLI selects its config automatically and prints JSON containing the shape,
config, VGPR count, spills, average microseconds, TFLOPS, buffer-set count,
timed-launch count, and graph-replay count. No config arguments are needed.

For a native host installation, run from the AITER repository root:

```bash
gpu-lock python3 -m op_tests.op_benchmarks.triton.bench_gemm_warp_pipeline \
  --shape 512 6144 7168 --warmup 20 --repeats 100 --seed 0
```

## Verify correctness

`--verify-only` launches WP and checks an independent FP64 reference, without
performing the rotating-buffer benchmark:

```bash
gpu-lock docker exec -w /app/aiter "$WP_CONTAINER" \
  python3 -m op_tests.op_benchmarks.triton.bench_gemm_warp_pipeline \
  --shape 512 6144 7168 --seed 0 --verify-only
```

The reference applies the E8M0 scales to the FP8 inputs in FP64, computes GEMM
in chunks of 1024 output columns, and rounds to BF16. It uses the stock
benchmark's BF16 tolerance: `rtol=0.016`,
`atol=5e-5 * max(abs(FP64_reference))`. This is a numerical tolerance check,
not a requirement for bitwise agreement with FlyDSL or previous Gluon.

To verify and benchmark in the same process, append `--check`:

```bash
gpu-lock docker exec -w /app/aiter "$WP_CONTAINER" \
  python3 -m op_tests.op_benchmarks.triton.bench_gemm_warp_pipeline \
  --shape 512 6144 7168 --warmup 20 --repeats 100 --seed 0 --check
```

`--check` also requires every rotated output to match the warmup output exactly.
Reference computation occurs before timing and can affect machine/cache state.
For the timing-only comparison protocol, run verification separately, then run
without `--check`. FP64 verification of the largest shapes can take appreciably
longer than the GEMM benchmark itself.

## Benchmark all 12 shapes

Run this Bash loop on the host. It acquires and releases `gpu-lock` for each
shape, and writes logs to a new directory. Each process uses 100 newly prepared
buffer sets and one timed graph replay.

```bash
WP_CONTAINER=aiter-triton-main-compare
WP_RESULTS=$(mktemp -d /tmp/wp-bench.XXXXXX)

for M in 512 16384; do
  for NK in 6144,7168 7168,3072 8192,1536 2048,7168 65536,1536 7168,16384; do
    N=${NK%,*}
    K=${NK#*,}
    if ! gpu-lock docker exec -w /app/aiter "$WP_CONTAINER" \
      python3 -m op_tests.op_benchmarks.triton.bench_gemm_warp_pipeline \
      --shape "$M" "$N" "$K" --warmup 20 --repeats 100 --seed 0 \
      > "$WP_RESULTS/$M-$N-$K.log" 2>&1; then
      cat "$WP_RESULTS/$M-$N-$K.log"
      exit 1
    fi
    cat "$WP_RESULTS/$M-$N-$K.log"
  done
done
printf 'Results: %s\n' "$WP_RESULTS"
```

For **verification of all shapes**, use the same loop with the CLI arguments
`--shape "$M" "$N" "$K" --seed 0 --verify-only` instead of the timing arguments.
For verification plus timing and rotated-output checks, append `--check` to the
benchmark invocation. In both cases keep `gpu-lock` inside the loop.

## Python API

For application integration, import the public API and prepare once:

```python
from aiter.ops.triton.gemm.basic.gemm_warp_pipeline_cdna5 import (
    prepare_gemm_warp_pipeline,
)

# a[M,K], w[N,K]: contiguous GPU float8_e4m3fn tensors.
# sa[M,K//128], sw[N//128,K//128]: contiguous GPU uint8 E8M0 codes.
prepared = prepare_gemm_warp_pipeline(a, w, sa, sw)
y = prepared()  # GEMM only; returns the preallocated BF16 output.
```

Run scripts containing these launches under `gpu-lock`. Calling the same
`prepared` object repeatedly reuses the same buffers; rotation requires separate
prepared objects, as implemented in the benchmark CLI. Explicit configurations
can be passed through the preparation API with `tile`, `cluster`, and
`flat_pipeline` overrides; the CLI deliberately uses the selected defaults.
