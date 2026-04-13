# Radix TopK Kernel Optimization (Baseline -> V4)

## Test Environment

- GPU: AMD Instinct MI300X (gfx942), 80 CU
- ROCm 6.4.2, hipcc -O3 -std=c++17
- Parameters: LEN=60000, K=2048, BATCH=1, float32, IS_LARGEST=true

## Latency Results

| Version  | Prefill (us) | Decode (us) | vs Baseline |
|----------|-------------|-------------|-------------|
| Baseline |  73.77      |  74.21      | 1.00x       |
| V1       |  55.26      |  55.51      | 1.34x       |
| V2       |  41.84      |  41.18      | 1.76x       |
| V3       |  30.20      |  28.96      | 2.44x       |
| V4       |  26.21      |  26.20      | 2.81x       |

All versions pass correctness verification (Value set match: PASS).

## Optimization Journey

### Baseline (`baselline_topk_per_row_kernels.cu`)
- BPP=11 (11+11+10 = 32 bits, 3 passes)
- One-block path + multi-block path, grid_dim=15
- Per-pass kernel launch (host for loop, 3 launches)
- `__threadfence()` after histogram flush
- Latency: ~74 us

### V1 (`topk_bothPD_mulblocks_v1.cu`)
- Multi-block only, grid_dim capped at 8
- Per-pass kernel launch (host for loop, 3 launches)
- Single histogram buffer, last-block clears between passes
- atomicInc + last-block election for inter-block sync
- `__threadfence()` after histogram flush
- **Optimization**: Removed one-block path overhead, tuned grid_dim
- Latency: ~55 us (1.34x)

### V2 (`topk_bothPD_mulblocks_v2.cu`)
- **Persistent kernel**: single launch covers all 3 passes (eliminates launch overhead)
- Per-pass independent histogram buffers (`hist_base + pass * num_buckets`)
- Eliminates Barrier 2 (no need to clear histogram between passes)
- Total barriers reduced from 5 to 3 (for BPP=11)
- Still uses `__threadfence()` after histogram flush
- **Optimization**: Persistent kernel + per-pass histogram -> fewer barriers
- Latency: ~41 us (1.76x)

### V3 (`topk_bothPD_mulblocks_v3.cu`)
- Removes all 3 `__threadfence()` calls
- MI300X atomicAdd to global memory is device-scope (direct L2 operation), fence is redundant
- Uses `__ATOMIC_RELEASE` / `__ATOMIC_ACQUIRE` for barrier synchronization
- **Optimization**: Eliminate unnecessary threadfence (~12 us savings)
- Latency: ~30 us (2.44x)

### V4 (`topk_bothPD_mulblocks_v4.cu`)
- BPP=10 (10+10+10+2 = 32 bits, 4 passes, but pass 3 is trivial with only 4 buckets)
- grid_dim capped at 15 instead of 8 (more parallelism)
- 1024 buckets per pass -> better load balance, smaller histogram
- **Optimization**: Smaller BPP + higher grid_dim -> more parallelism
- Latency: ~26 us (2.81x)

## Build & Run

```bash
# Compile (pick one)
hipcc -O3 -std=c++17 test_baseline.cu -o test_baseline --offload-arch=gfx942
hipcc -O3 -std=c++17 test_topk_v1.cu  -o test_topk_v1  --offload-arch=gfx942
hipcc -O3 -std=c++17 test_topk_v2.cu  -o test_topk_v2  --offload-arch=gfx942
hipcc -O3 -std=c++17 test_topk_v3.cu  -o test_topk_v3  --offload-arch=gfx942
hipcc -O3 -std=c++17 test_topk_v4.cu  -o test_topk_v4  --offload-arch=gfx942

# Run
./test_baseline 60000 2048 1
./test_topk_v1  60000 2048 1
./test_topk_v2  60000 2048 1
./test_topk_v3  60000 2048 1
./test_topk_v4  60000 2048 1
```

## File Structure

```
baselline_topk_per_row_kernels.cu  # Baseline kernel
topk_bothPD_mulblocks_v1.cu        # V1 kernel
topk_bothPD_mulblocks_v2.cu        # V2 kernel
topk_bothPD_mulblocks_v3.cu        # V3 kernel
topk_bothPD_mulblocks_v4.cu        # V4 kernel
test_baseline.cu                   # Baseline test harness
test_topk_v1.cu                    # V1 test harness
test_topk_v2.cu                    # V2 test harness
test_topk_v3.cu                    # V3 test harness
test_topk_v4.cu                    # V4 test harness
```
