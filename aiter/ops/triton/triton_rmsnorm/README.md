# triton_rmsnorm — fused all-reduce + residual add + RMSNorm for gfx950

`TlLamport2RMSNorm` computes, over the ranks of a group:

```
residual_out = all_reduce(x) + residual        (fp32 sum, rounded once)
out          = RMSNorm(residual_out) * weight  (bf16, or static per-tensor fp8)
```

It is written in plain Triton (3.4+) and runs as 2 kernels:

1. **All-reduce + residual add:** the Lamport protocol of aiter's custom all-reduce (CAR).
   - Every rank pushes into the peers' uncached inboxes. The data itself is the arrival signal: inboxes start as 0xFFFF, and senders turn 0xFFFF halves into 0x7FFF.
   - 3 rotating buffers replace barriers. Each program owns fixed inbox positions and one round counter in device memory, so HIP graphs work with no buffer registration.
   - **one-shot** (≤ 4 rows by default): every rank pushes its whole x to every peer.
   - **two-shot**: a reduce-scatter push, then an all-gather push.
   - **Wave split:** a `[NGPU, 64-word]` tile with `num_warps = NGPU`, so wave g handles GPU g: one push, one poll and one re-arm per lane.
2. **Norm:** a local RMSNorm of `residual_out`, one program per row.

All ranks get the same bits.

```python
from aiter.ops.triton.triton_rmsnorm import TlLamport2RMSNorm

op = TlLamport2RMSNorm(rank, world, hidden=8192, max_rows=2048)
op.register(gloo_group)                                   # collective, eager
out, residual_out = op(x, residual, weight, eps)          # bf16 out
out, residual_out = op(x, residual, weight, eps, scale)   # fp8 e4m3fn out
```

Requirements:
- bf16 input; hidden a power of 2 from 512 to 16384; up to `max_rows` rows; 2, 4 or 8 GPUs.
- The inbox is about 193 MB per GPU at H=8192 with 2048 rows.

| file | |
|---|---|
| `tl_lamport2.py` | the op and both kernels |
| `csrc/lamport_ipc.cpp` | uncached IPC inbox (`hipExtMallocWithFlags` + hipIpc), JIT-built into `~/.aiter/triton_rmsnorm/torch-<version>/` (`TRITON_RMSNORM_BUILD_DIR` overrides) |
| `test_tl_lamport2.py` | correctness: one-shot and two-shot, uneven parts, modes alternating back to back, graph replay with new data; bf16 and fp8 |
| `bench_lam2.py` | graph-replay latency vs CAR + norm, aiter fused and RCCL + norm |

## Results (2026-11-17)

Setup:
- 4× MI355X (gfx950), Triton 3.8.0, torch 2.13.0 + ROCm 7.14.
- H=8192, timings in µs/call: 10 calls per graph × 50 replays, best of 3, slowest rank.
- One row count per process.
- fp8: lam2 and CAR + norm use static per-tensor scaling; aiter fused uses per-token scaling.

| rows | lam2 bf16 | CAR + norm bf16 | aiter fused bf16 | lam2 fp8 | CAR + norm fp8 | aiter fused fp8 |
|---|---|---|---|---|---|---|
| 1 | **5.62** | 6.32 | 7.57 | **6.01** | 8.00 | 9.40 |
| 4 | **6.68** | 7.04 | 7.82 | **6.95** | 8.77 | 9.90 |
| 16 | **8.54** | 8.94 | 10.39 | **8.78** | 10.79 | 12.65 |
| 64 | **15.47** | 15.67 | 16.45 | **15.62** | 17.54 | 19.02 |
| 128 | **24.71** | 25.84 | 28.38 | **24.77** | 27.81 | 31.04 |
| 256 | **43.00** | 43.48 | 47.77 | **43.15** | 45.57 | 50.67 |
| 512 | **79.26** | 82.04 | 86.88 | **79.49** | 84.44 | 90.87 |
| 1024 | **150.64** | 157.03 | 161.90 | **151.67** | 162.33 | 167.44 |
| 2048 | **293.68** | 307.96 | 312.40 | **294.90** | 315.38 | 321.36 |

Breakdown, bf16:

| rows | lam2 AR | CAR AR | lam2 norm | aiter norm |
|---|---|---|---|---|
| 1 | **3.69** | 3.86 | **2.14** | 2.52 |
| 16 | **6.46** | 6.49 | **2.28** | 2.65 |
| 64 | **13.05** | 13.10 | **2.35** | 2.82 |
| 256 | 39.83 | **39.01** | **2.86** | 3.51 |
| 512 | **74.92** | 75.12 | **3.79** | 5.87 |
| 2048 | 281.41 | **281.15** | **10.66** | 23.29 |

- **All-reduce:** at parity with CAR.
- **Norm:** faster than `rmsnorm2d_fwd_with_add`. The margin grows with rows, to about 2.2× at 2048.
- **Tile size:** the 64-word tile replaced a 256-word one (`TL2_WS_BLK=256` restores it). It gained 0.5–3.9 µs on the full op at every row count except 64, where it lost 0.5 µs.
- **vLLM:** in Llama TP=4 decode (vLLM 0.9.2, Triton 3.4), it was 2.3–3.0% faster end to end than native vLLM in bf16, and 6.1–7.0% faster with fp8 output.

## Rules found the hard way

- **Polls** are `cache_modifier=".cv"` loads. **Pushes and re-arms** are `".wt"` stores. Otherwise a poll can keep reading a stale cached line and deadlock.
- **Poll loads are unmasked** (slots are allocated in full), and the tail mask is applied only in the exit test. A masked poll loop compiled to irreducible control flow, and some lanes were never re-read. A barrier runs in each poll iteration, because the waves share the reduction scratch.
- **The tile size (BLK) is fixed per mode.** Round counters and inbox positions are tied to it.
- **Triton's AMD buffer-ops pass is off** for these kernels (Triton ≥ 3.8). With it on, polls became buffer loads that never saw the peers' pushes.
- **Summation order:** the ranks are added as a tree (`(x0+x2)+(x1+x3)` at 4 GPUs), not in rank order. Where values cancel, the result can differ from a rank-order reference by more than 1 ulp. The test therefore uses the fp32 error bound, and checks that all ranks agree bit for bit.

## Run

```bash
cd aiter/ops/triton/triton_rmsnorm && export HIP_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 test_tl_lamport2.py                       # PASS
ROWS=1,3,5,7,33,100,1027,2 torchrun --nproc_per_node=4 test_tl_lamport2.py
PROVIDERS=lam2 ROWS=1,64,512 torchrun --nproc_per_node=4 bench_lam2.py
PROVIDERS=car,car_fused ROWS=1,64,512 python3 bench_lam2.py
```

Add `FP8=1` for fp8 output.
