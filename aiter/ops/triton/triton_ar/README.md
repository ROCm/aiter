# triton_ar — Triton all-reduce for gfx950

A plain-Triton (3.4+) all-reduce (sum, bf16/fp16) using the Lamport push protocol of
aiter CAR, from the wave-split all-reduce kernels of the fused all-reduce + RMSNorm work.

| file | |
|---|---|
| `triton_ar.py` | `TritonAllReduce(rank, nGPU, max_bytes, one_shot_bytes=64K)`; `register(gloo)`; `ar(x, out=None)` |
| `csrc/lamport_ipc.cpp` | uncached IPC inbox; JIT-built into `~/.aiter/triton_ar/torch-<version>/` (`TRITON_AR_BUILD_DIR` overrides) |
| `test_triton_ar.py` | correctness: sizes across one-shot/two-shot and uneven parts, eager and graph replay, bf16 and fp16 |
| `bench_ar.py` | graph-replay latency: `triton` (torchrun + gloo), `rccl` and `car` (aiter path, separate process) |
| `test_fp8.py` | fp8 all-gather mode correctness |

Design:
- **one-shot** (≤ 64 KB): push the whole tensor to every peer, then sum.
- **two-shot**: reduce-scatter push, then all-gather push.
- 3 rotating inboxes and per-program round counters, so no barrier and it is graph-safe.
- `[NGPU, BLK]` tile with `num_warps = NGPU`: wave g handles GPU g. Ranks are summed in a fixed order, so all ranks get the same bits.
- BLK = 64 words for both modes (from a sweep).

## Results (2026-11-16)

Setup: `pytorch-vllm-dist-rms-image` (Triton 3.8, ROCm 7.14), 4× MI355X, bf16. Each number is µs per call: 10 calls per graph × 50 replays, best of 3, slowest rank. The number in brackets is algbw in GB/s.

| size | RCCL | aiter CAR | triton_ar |
|---|---|---|---|
| 8K | 12.11 (0.7) | 3.75 (2.2) | **3.63** (2.3) |
| 16K | 12.58 (1.3) | 3.89 (4.2) | **3.66** (4.5) |
| 32K | 12.94 (2.5) | 4.05 (8.1) | **3.90** (8.4) |
| 64K | 13.12 (5.0) | 4.59 (14.3) | **4.54** (14.4) |
| 128K | 25.39 (5.2) | 5.49 (23.9) | **5.41** (24.2) |
| 256K | 25.88 (10.1) | 6.45 (40.6) | **6.35** (41.3) |
| 512K | 26.89 (19.5) | **8.34** (62.9) | 8.41 (62.3) |
| 1M | 32.87 (31.9) | **12.86** (81.5) | 13.18 (79.5) |
| 2M | 40.17 (52.2) | 22.41 (93.6) | **21.79** (96.2) |
| 4M | 58.38 (71.8) | **39.09** (107.3) | 39.78 (105.4) |
| 8M | 92.50 (90.7) | 74.94 (111.9) | **74.31** (112.9) |
| 16M | 162.64 (103.2) | 143.08 (117.3) | **142.83** (117.5) |
| 32M | 303.93 (110.4) | 280.79 (119.5) | **280.13** (119.8) |
| 64M | 597.10 (112.4) | 557.70 (120.3) | **556.07** (120.7) |

- **vs aiter CAR:** triton_ar is equal or faster everywhere except 512K–1M and 4M, where it is up to 2.5% slower. It is about 3–6% faster at ≤ 32K.
- **vs RCCL:** RCCL is 3–4× slower at small sizes and about 7% slower at 64M.

Note: aiter's `tensor_model_parallel_all_reduce` may hand very large tensors to RCCL internally. The CAR column is whatever that call does.

### Sweeps

- **One-shot/two-shot switch:** 64 KB is best. Two-shot costs +1.3 µs at 8K; one-shot costs +10 µs at 1M.
- **BLK at 1 MB:** 64 / 128 / 256 / 512 give 13.1 / 12.8 / 12.6 / 17.7 µs.
- **BLK for small and large sizes:** 64 is best for one-shot, and for two-shot at small and large sizes. The 1–2 MB band prefers 256, but see bug 3.

## fp8 all-gather mode (2026-11-16)

`TritonAllReduce(..., fp8=True)` and `ar(x, fp8=True)` give a lossy two-shot, the same scheme as aiter CAR's `open_fp8_quant`:
- **phase 1 (reduce-scatter):** pushed at full precision.
- **phase 2 (all-gather):** pushed as e4m3fn, with one fp32 scale (amax/448) per 128 values. That is one 64-word tile, and the scale word sits in the inbox after the data.
- **Owner rank:** writes the dequantized values too, so all ranks get the same bits. In aiter, the owner keeps the exact sum.
- **Dtypes and sizes:** bf16 and fp16, for sizes above `one_shot_bytes`. aiter quantizes only fp16 with numel ≥ 262144 (512 KB); below that it runs plain CAR.

`test_fp8.py` checks:
- the result against the fp32 sum quantized the same way, with 1 fp8 step of slack;
- that all ranks agree;
- eager calls and graph replay.

The error is max|err|/max|sum| ≈ 3% for both bf16 and fp16.

Results: fp16 inputs (all providers), µs/call with algbw in GB/s. Same harness as above.

| size | RCCL | CAR | CAR fp8 | triton | triton fp8 |
|---|---|---|---|---|---|
| 8K | 12.46 | 3.75 | 3.78 | **3.64** | 3.67 |
| 64K | 12.97 | 4.63 | 4.65 | **4.49** | 4.50 |
| 128K | 25.15 | 5.54 | 5.53 | **5.47** | 8.16 |
| 512K | 26.77 | 8.35 | 10.11 | **8.34** | 10.40 |
| 1M | 32.49 | **12.69** | 13.79 | 12.93 | 14.88 |
| 2M | 39.86 | 22.45 | **20.47** | 21.80 | 23.12 |
| 4M | 58.13 | 39.20 | **33.73** | 39.61 | 41.71 |
| 8M | 92.20 | 75.10 | **62.18** (134.9) | 74.40 | 75.60 (111.0) |
| 16M | 162.29 | 143.24 | **114.80** (146.1) | 143.36 | 137.11 (122.4) |
| 32M | 304.04 | 281.54 | **221.00** (151.8) | 281.57 | 261.37 (128.4) |
| 64M | 597.15 | 556.34 | **434.35** (154.5) | 557.61 | 512.87 (130.9) |

- Below 8K–64K both fp8 modes fall back to the full-precision path (one-shot), so they match plain.
- **CAR fp8** wins from 2M up: 22% faster than plain CAR at 64M, close to the 25% the traffic cut allows (phase 2 halves, so traffic is 0.75×).
- **triton fp8** only pays off from 16M (8% faster at 64M). Below that it is slower than plain triton.
  - The kernel uses 64-word tiles. The fp8 phase uses half the lanes, and there are several LDS reshapes.
  - It has a second inbox read after the polls (see bug 4).
  - Next step: several 128-value groups per program.

## Bugs found (all fixed; also relevant to tl_lamport2)

1. **Masked poll loads miscompile.**
   - **Cause:** the poll used a tail mask (`gpos < nw`) on an uneven size. With it, the poll loop compiled to per-dword branches with irreducible control flow. Some dwords were never re-read, the loop exited, and the raw 0xFFFF sentinel was copied to the output.
   - **Symptom:** NaN at the tail on the peer ranks, intermittent, only when the two-shot parts are uneven.
   - **Fix:** poll loads are unmasked (slots are always allocated in full), and the mask is applied only to the exit test. This also gives straight-line `dwordx4` loads.
2. **fp16 reference artifact.** `tl.sum` adds the ranks as a tree, and an fp16 sum in fp32 is not always exact. The test now allows 1 ulp against the fp32 sum. All ranks must still agree bit for bit.
3. **BLK must be fixed per mode.** A program owns fixed inbox positions and one round counter. Two-shot calls with different BLKs share the counters, so the rounds go out of step and the values come out wrong.

4. **fp8 kernel bugs.**
   - **Deadlock:** a 1-D `[NGPU]` scale vector in the poll deadlocked. The scale is now polled in the `[NGPU, 32]` tile.
   - **Stale values after the loop:** the values carried out of the poll loops were stale (whole NaN tiles), even though the loops had seen the data arrive. The inbox is now read once more after the polls.
   - **Debug aid:** `TAR_MAXIT` caps the polls and counts missing words, but the cap itself can hang. Debug only.

Also kept from tl_lamport2:
- polls use `.cv` loads;
- pushes and re-arms use `.wt` stores;
- AMD buffer ops are off;
- a `tl.debug_barrier()` runs in each poll iteration (the waves share the reduction scratch).

## Run

```bash
cd aiter/ops/triton/triton_ar && export HIP_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 test_triton_ar.py
PROVIDERS=triton torchrun --nproc_per_node=4 bench_ar.py
PROVIDERS=rccl,car python3 bench_ar.py
```

Env: `SIZES` (bytes), `DTYPE` (bf16/fp16), `PROVIDERS` (triton, triton_fp8 | rccl, car, car_fp8), `ONE_SHOT_BYTES`, `TAR_BLK1`, `TAR_BLK2`, `CALLS`, `REPLAYS`, `TRIALS`.


## Host run (2026-11-16, ~/triton-lang-fork/.venv: Triton 3.9, torch 2.14, ROCm 7.1)


µs/call; container numbers in brackets.

| size | RCCL bf16 | CAR bf16 | triton bf16 | triton fp16 | triton fp8 (fp16 in) |
|---|---|---|---|---|---|
| 8K | 15.40 [12.11] | 4.16 [3.75] | 4.13 [3.63] | 4.13 | 4.06 |
| 64K | 15.94 [13.12] | 4.93 [4.59] | 4.92 [4.54] | 4.89 | 4.91 |
| 256K | 29.58 [25.88] | 6.81 [6.45] | 6.78 [6.37] | 6.84 | 9.00 |
| 1M | 41.24 [32.87] | 13.30 [12.86] | 13.68 [13.18] | 13.55 | 15.56 |
| 4M | – | – | 40.08 [39.78] | 40.06 | 42.24 |
| 16M | – | – | 143.93 [142.83] | 143.90 | 138.02 |
| 64M | – | – | 558.16 [556.07] | 558.56 | 513.77 |

- Everything is slower on the host than in the container: about 0.4 µs at small sizes, about equal from 4M up.
- On the host, triton ≈ CAR (±0.4 µs).
- **aiter on the host is unreliable:** sizes ≥ 2M (bf16) and every fp16 run with `car_fp8` hung until the timeout. The first full-list run lost its rendezvous store connection at startup. One fp16 triton run segfaulted on rank 0 and passed on retry.
