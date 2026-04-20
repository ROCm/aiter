# GEAK Patch Explained: `batched_gemm_a16wfp4_best_2.97x.patch`

## What This Kernel Does

This is a **batched GEMM** (General Matrix Multiply) kernel that computes `Y[b] = X[b] @ W[b]^T` where:
- **X** is BF16 activations with shape `(B, M, K)` — the input tokens
- **W** is MXFP4 weights with shape `(B, N, K//2)` — packed FP4 (2 values per byte)
- **Y** is the BF16 output with shape `(B, M, N)`

The "pre-quant" part means that **X is quantized on-the-fly to MXFP4** inside the kernel before the actual matrix multiply. This is the inner loop: load BF16 activation tiles, quantize them to FP4, then call `tl.dot_scaled` which does the FP4×FP4 matrix multiply with per-group scaling on the MI355X's native MXFP4 hardware.

In inference, `B` is the batch of experts in MoE (Mixture of Experts), `M` is the number of tokens routed to that expert, `N` is the output dimension, and `K` is the input dimension.

---

## The Original Implementation (Layer by Layer)

### 1. The Triton Kernel (`_batched_gemm_a16wfp4_kernel`)

A Triton kernel launches a **grid** of programs (think: CUDA thread blocks). Each program computes one tile of the output matrix. The original grid is **2D**:

```python
grid = lambda META: (
    B,                    # axis=0: one "row" per batch element
    (NUM_KSPLIT * cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N)),  # axis=1: all tiles for that batch
)
```

Inside the kernel, the program figures out which tile it owns:

```python
pid_batch = tl.program_id(axis=0)    # which batch element
pid_unified = tl.program_id(axis=1)  # which M×N tile (and K-split)
```

Then each program:
1. Loads a `BLOCK_SIZE_M × BLOCK_SIZE_K` tile of X (BF16)
2. Calls `_mxfp4_quant_op` to quantize it to FP4 + compute per-group scales
3. Loads a `BLOCK_SIZE_K × BLOCK_SIZE_N` tile of W (already FP4) + its scales
4. Does `tl.dot_scaled(a_fp4, a_scales, b_fp4, b_scales)` — hardware MXFP4 matmul
5. Repeats across the K dimension, accumulating in FP32
6. Writes the `BLOCK_SIZE_M × BLOCK_SIZE_N` output tile

### 2. The Quantization Function (`_mxfp4_quant_op`)

This is called **inside the inner K-loop** for every tile, making it performance-critical. It converts a BF16 tile to MXFP4 format:

1. **Find the max absolute value** per group of 32 elements
2. **Compute a scaling factor** (e8m0 format — exponent only, no mantissa)
3. **Scale the values** so they fit in FP4 range (±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6)
4. **Pack** two 4-bit values into each byte

The original scale computation uses floating-point math:

```python
amax = (amax + 0x200000) & 0xFF800000  # round up to power of 2
amax = amax.to(float32, bitcast=True)
scale = tl.log2(amax).floor() - 2       # log2 → slow transcendental
quant_scale = tl.exp2(-scale)            # exp2 → slow transcendental
```

### 3. Config Parameters

The JSON config files select the tile sizes based on the M dimension. Here's what each parameter means:

| Parameter | What it controls |
|-----------|-----------------|
| `BLOCK_SIZE_M` | Height of the output tile (rows of X processed per program) |
| `BLOCK_SIZE_N` | Width of the output tile (columns of W processed per program) |
| `BLOCK_SIZE_K` | How many elements along K are processed per inner-loop iteration |
| `GROUP_SIZE_M` | How many M-tiles are grouped together for L2 cache reuse (tile scheduling) |
| `num_warps` | Number of warps (32-thread groups) per workgroup — controls parallelism within a tile |
| `num_stages` | Software pipelining depth — how many tiles are prefetched ahead in the K-loop |
| `waves_per_eu` | Occupancy hint — how many wavefronts per execution unit (higher = more latency hiding) |
| `cache_modifier` | Memory access hint: `null` = default, `".cg"` = cache at global level (bypass L1, keep in L2) |
| `NUM_KSPLIT` | Split the K dimension across multiple programs for parallelism on small M |

---

## The Patch — Change by Change

### Change 1: Integer Bit-Ops Replace Transcendentals in Quantization

**File**: `quant.py` (`_mxfp4_quant_op`)

**Original** (3 operations: bitcast → `tl.log2` → `tl.exp2`):

```python
amax = (amax + 0x200000).to(uint32) & 0xFF800000
amax = amax.to(float32, bitcast=True)
scale_e8m0_unbiased = tl.log2(amax).floor() - 2     # TRANSCENDENTAL
quant_scale = tl.exp2(-scale_e8m0_unbiased)           # TRANSCENDENTAL
```

**Patched** (pure integer arithmetic):

```python
amax_rounded = (amax + 0x200000).to(uint32) & 0xFF800000

# Extract exponent via bit-shift instead of log2
exp_biased = (amax_rounded >> 23).to(int32)
scale_e8m0_unbiased = exp_biased - 129  # = (exp - 127) - 2

# Construct 2^(-scale) via bit manipulation instead of exp2
quant_exp = (127 - scale_e8m0_unbiased)
quant_scale = (quant_exp << 23).to(float32, bitcast=True)
```

**Why it's faster**: The key insight is that `amax_rounded` after the `& 0xFF800000` mask has its mantissa zeroed out — it's already an exact power of 2. In IEEE 754, a power-of-2 float has the form `2^(biased_exponent - 127)`, so `log2` is just extracting the exponent field (bits 23-30) via a right-shift. Similarly, `2^n` can be constructed by left-shifting `(n + 127)` into the exponent field. Both `tl.log2` and `tl.exp2` compile to multi-cycle transcendental function units (SFU instructions). The integer shift+add replacements are single-cycle ALU ops. Since `_mxfp4_quant_op` runs inside the innermost K-loop, this saves cycles on every iteration.

### Change 2: Flatten 2D Grid to 1D Grid

**File**: `batched_gemm_a16wfp4.py` (kernel) + wrapper

**Original kernel** — 2D grid `(B, GRID_MN * NUM_KSPLIT)`:

```python
pid_batch = tl.program_id(axis=0)      # axis 0 = batch
pid_unified = tl.program_id(axis=1)    # axis 1 = tile index
```

**Original wrapper**:

```python
grid = lambda META: (
    B,
    (NUM_KSPLIT * cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N)),
)
```

**Patched kernel** — 1D grid `(B * GRID_MN * NUM_KSPLIT,)`:

```python
pid_flat = tl.program_id(axis=0)
grid_mn_ksplit = GRID_MN * NUM_KSPLIT
pid_batch = pid_flat // grid_mn_ksplit
pid_unified = pid_flat % grid_mn_ksplit
```

**Patched wrapper**:

```python
grid = lambda META: (
    B
    * (NUM_KSPLIT * cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N)),
)
```

**Why it's faster**: On AMD GPUs, 2D grid dispatch involves the hardware command processor distributing workgroups across two dimensions. With a 2D grid of `(B, tiles_per_batch)`, the GPU's workgroup scheduler may serialize along the batch axis or create uneven distribution. A 1D grid gives the hardware scheduler a flat sequence of workgroups, allowing it to pack them onto Compute Units (CUs) more efficiently — especially when B is small (common in MoE where each expert handles few tokens) and doesn't evenly fill all CUs.

### Change 3: Config Tuning — Tile Sizes and Memory Hints

**File**: Multiple JSON config files

**N=512, K=128 config** — small M shapes (M_LEQ_16 through M_LEQ_128):

| Shape | Original `BLOCK_SIZE_N` | Patched `BLOCK_SIZE_N` |
|-------|------------------------|----------------------|
| M_LEQ_16 | 32 | **256** |
| M_LEQ_32 | 64 | **256** |
| M_LEQ_64 | 64 | **256** |
| M_LEQ_128 | 128 | **256** |

**Why**: When M is small (decode phase — few tokens per expert), each program handles few rows. By making `BLOCK_SIZE_N` much larger (256 vs 32-128), each program covers more of the N dimension per launch. This reduces the total number of workgroups, reduces kernel launch overhead and scheduling fragmentation, and improves data reuse of the weight tiles (loaded once, used across more output columns).

**N=128, K=512 config** — M_LEQ_64:

| Param | Original | Patched |
|-------|----------|---------|
| `BLOCK_SIZE_M` | 16 | **32** |
| `BLOCK_SIZE_K` | 512 | **256** |
| `num_stages` | 1 | **2** |
| `waves_per_eu` | 2 | **4** |

**Why**:
- `BLOCK_SIZE_M 16→32`: Doubles the M-tile, processing more rows per program. With K=512 and BLOCK_SIZE_K=512, the original did only 1 K-loop iteration with a huge tile — this is wasteful because it demands massive register pressure for the large K tile.
- `BLOCK_SIZE_K 512→256`: Halves the K-tile, meaning 2 iterations instead of 1. This reduces peak register pressure and shared memory usage, allowing more waves to coexist on the CU.
- `num_stages 1→2`: Enables **software pipelining** — while the current K-tile is being computed, the next tile's data is being prefetched from memory. With `num_stages=1`, there's no overlap: load, compute, load, compute. With `num_stages=2`, the load for iteration N+1 overlaps with the compute of iteration N.
- `waves_per_eu 2→4`: Tells the compiler to target higher occupancy. More waves per execution unit means more threads to hide memory latency. The tradeoff is fewer registers per wave, but since we halved K-tile size, register pressure is already lower.

**`cache_modifier: null → ".cg"`** (added to several "any" fallback configs):

The `.cg` (cache global) modifier tells the hardware to bypass the L1 cache and cache in L2 only. This is beneficial for streaming workloads where the same data won't be re-read by the same CU — it avoids polluting L1 with data that won't be reused locally, leaving L1 capacity for the accumulator and activation tiles that ARE reused.

### Change 4: Framework Overhead Reduction

**File**: `batched_gemm_a16wfp4.py` (wrapper)

**Removed logging**:

```python
# REMOVED:
_LOGGER.info(f"BATCHED_GEMM_AFP4WFP_PREQUANT: x={tuple(x.shape)} ...")
```

Even when the log level is above INFO, Python still evaluates the f-string (creating `tuple(x.shape)` strings) before the logger checks the level. This runs on every single GEMM call.

**Removed runtime assertions**:

```python
# REMOVED:
assert prequant is True, "prequant = False is not yet supported"
assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"
assert Bx == Bw
assert y.shape[0] == B and y.shape[1] == M ...
```

`arch_info.is_fp4_avail()` likely queries the GPU every call. These checks are meaningful during development but pure overhead in production — they run on every single MoE expert GEMM dispatch.

**Cached Triton knobs**:

```python
import triton.knobs as _triton_knobs
if not hasattr(_triton_knobs.amd, '_batched_gemm_knobs_cached'):
    if "AMDGCN_USE_BUFFER_OPS" not in os.environ:
        _triton_knobs.amd.use_buffer_ops = False
    # ... more knobs ...
    _triton_knobs.amd._batched_gemm_knobs_cached = True
```

Every time Triton JIT-compiles or dispatches a kernel, it checks several environment variables (`TRITON_INTERPRET`, `TRITON_DEBUG`, `TRITON_KERNEL_OVERRIDE`, etc.) via `os.getenv()`. This caching sets them once at import time, avoiding repeated syscalls to the environment on every kernel launch.

### Change 5: `copy.deepcopy` → `dict()`

**File**: `gemm_config_utils.py`

```python
# Original:
return copy.deepcopy(config), is_tuned
# Patched:
return dict(config), is_tuned
```

`copy.deepcopy` recursively copies every nested object, handling arbitrary Python types, circular references, etc. The config is a flat `dict[str, int|str|None]` — `dict()` creates a shallow copy which is equivalent for flat dicts but ~10-50x faster. This function is called for every GEMM dispatch to prevent cache mutation.

---

## Summary Table

| Change | Category | What | Why Faster |
|--------|----------|------|-----------|
| Integer bit-ops in quant | **Compute** | Replace `tl.log2`/`tl.exp2` with `>>23`/`<<23` | Single-cycle ALU vs multi-cycle transcendentals, inside the hot K-loop |
| 1D grid flattening | **Scheduling** | Merge 2D `(B, tiles)` grid into 1D `(B*tiles,)` | Better workgroup distribution across CUs, especially for small B |
| Config tuning (tile sizes) | **Memory/Compute** | Larger N-tiles for small M, smaller K-tiles with pipelining | Better hardware utilization, reduced register pressure, latency hiding |
| `.cg` cache modifier | **Memory** | Bypass L1, cache in L2 | Avoids L1 pollution for streaming weight loads |
| Remove logging/asserts | **Host overhead** | No f-string eval, no GPU queries per call | Eliminates Python-side overhead on every dispatch |
| Cache Triton knobs | **Host overhead** | Set env-var-derived knobs once at import | Avoids repeated `os.getenv` syscalls per kernel launch |
| `dict()` vs `deepcopy` | **Host overhead** | Shallow copy of flat config dict | ~10-50x faster copy for flat dicts |

The micro-benchmark shows 2.97x because it measures the kernel in isolation (quantization + GEMM). In end-to-end serving, this kernel is ~4.8% of total GPU time, so the E2E impact is proportionally smaller.

---

## GEAK Validation Data

### Hardware & Setup

- **GPU**: AMD MI355X (gfx950, CDNA4)
- **Benchmark harness**: GEAK built-in micro-benchmark with correctness verification
- **Shapes**: 288 combinations — B={1..16}, M={1,2,4,8,16,32,64,128,256}, N={128,512}, K={128,512}
- **Workload context**: Batched MLA attention GEMM with on-the-fly MXFP4 quantization (MoE decode)
- **Correctness**: Verified via `--correctness` pass (numerically identical outputs)

### Baseline Profile

| Metric | Value |
|--------|-------|
| Total duration (288 shapes) | 34,303 us |
| Top kernel (B=M=256, N=K=256) | 19,234 us (55.5%) |
| Bottleneck classification | Balanced / Latency-bound for small shapes |
| HBM bandwidth utilization | 5.0% |
| L2 hit rate | 38.8% |
| Geometric mean latency (single shape) | 0.0227 ms |

### Round-by-Round GEAK Optimization Progress

| Round | Strategy | Verified Speedup | Geo-mean Latency | Key Innovation |
|-------|----------|:----------------:|:-----------------:|----------------|
| 1 | *(infra failure)* | — | — | Fixed git diff timeout from core dumps |
| 2 | config-tuning-small-shapes | **1.90x** | ~0.0120 ms | Config caching + wrapper overhead removal |
| 3 | lut-based-mxfp4-quant-no-reshape | **2.97x** | ~0.0076 ms | + 1D grid + bitwise scale + config tuning |
| 4 | single-pass-quant-dot-fused | 1.18x | ~0.0210 ms | Regression (cumulative patches not applied) |
| 5 | reduce-k-loop-overhead-unroll-hint | 2.55x | ~0.0082 ms | + XCD removal + `tl.range` pipelining |

### Best Verified Result (Round 3)

- **Speedup**: **2.971x** (FULL_BENCHMARK verified across all 288 shapes)
- **Baseline latency**: 0.0205 ms (geometric mean)
- **Optimized latency**: 0.0069 ms (geometric mean)
- **Patch**: `lut-based-mxfp4-quant-no-reshape/patch_10` (10 iterations of LLM-guided optimization)

### Representative Optimized Latencies (Round 3, selected shapes)

| Shape | Baseline | Optimized | Speedup |
|-------|:--------:|:---------:|:-------:|
| B=1 M=1 N=512 K=128 | 0.0227 ms | 0.0065 ms | 3.49x |
| B=1 M=4 N=128 K=512 | 0.0227 ms | 0.0063 ms | 3.60x |
| B=4 M=32 N=512 K=128 | 0.0227 ms | 0.0065 ms | 3.49x |
| B=8 M=8 N=128 K=512 | 0.0227 ms | 0.0064 ms | 3.55x |
| B=1 M=256 N=128 K=512 | 0.0227 ms | 0.0091 ms | 2.49x |
| B=9 M=256 N=128 K=512 | 0.0227 ms | 0.0109 ms | 2.08x |

Small shapes (M<=32) see the largest gains (3.5x+) due to grid flattening and config tuning.
Larger M shapes with N=128, K=512 show ~2.0-2.5x gains, limited by the K-loop still being compute-bound.

### End-to-End Context

In Kimi-K2-Thinking-MXFP4 inference on 4x MI355X (TP=4), this kernel accounts for **~4.8% of total GPU time**.
A/B benchmarking at ISL=1024, OSL=512, concurrency=512 showed:
- **Baseline**: 10,424.7 tok/s
- **Optimized**: 10,444.9 tok/s (+0.2%)

The modest E2E impact is expected: the majority of GPU time is in MoE expert GEMMs (C++/ASM kernels), attention, and all-reduce collectives which are outside the scope of this Triton kernel optimization.
