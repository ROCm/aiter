# FlyDSL MLA decode-reduce — benchmark report

**Target:** AMD Instinct MI300X (gfx942, CDNA3, 304 CUs, warp 64), ROCm 7.2.0.
**Kernel:** `compile_mla_reduce` in `aiter/ops/flydsl/kernels/mla_reduce.py` — native FlyDSL
port of the HIP `kn_mla_reduce_v1`. Stage-2 epilogue of split-KV MLA decode: LSE-weighted
online-softmax combine of per-split partials. **Pure reduction — no MFMA, HBM-bandwidth bound.**

Companion to `MLA-reduce-HIP-kernel-benchmark-report.md` (the baseline) and
`MLA-reduce-flydsl-vs-hip-comparison.md` (the head-to-head). Reproduction steps:
`flydsl_mla_reduce_REPRODUCE.md`.

---

## 1. Environment

| Component | Version / value |
|---|---|
| GPU | AMD Instinct MI300X (gfx942, 304 CUs, warp 64), HBM3 ~5.3 TB/s peak |
| Container | `rocm/ali-private:ubuntu22.04_rocm7.2.0.43_cp310_torch2.9.1_sglang_027c46b_aiter_eee0abe_qwen3_5_20260529` |
| ROCm / Python / PyTorch | 7.2.0 / 3.10.12 / 2.9.1+rocm7.2.0 |
| flydsl | 0.2.0 |

Harness: `op_tests/test_flydsl_mla_reduce.py`, 25 warmup / 100 timed iters, CUDA events.
BW = traffic model (`T·S·H·Dv·4` partial-O read + lse + final-O write) / latency.

---

## 2. Correctness

Full matrix — 3 shapes `(H,Dv) ∈ {(128,512),(16,512),(128,128)}` × {bf16, fp16} × 8 split
counts `{2,3,4,8,16,64,256,300}`, spanning all four tiers (simple / m64 / m256 / mlds):

**48 / 48 pass.** Output `max_abs_err ≤ 3.9e-3` (bf16) / `4.9e-4` (fp16);
LSE `max_abs_err ≤ 9.5e-7` (exact). Reference is a vectorized fp64 torch online-softmax.

> Large random-LSE inputs at very high tile counts can show a bf16 output spike (~7e0); this
> is a harness/bf16 degeneracy artifact reproduced identically by the HIP kernel, not a kernel
> bug. The `--matrix` sweep uses `tiles=4` and is unaffected.

---

## 3. Benchmark results

Shape **H=128, Dv=512, bf16** (GLM/DeepSeek decode shape) unless noted.

### 3.1 Important: the harness host-overhead floor

The standalone FlyDSL harness has a **~230 µs per-call Python host floor** (DLPack marshalling
+ launcher dispatch per timed iter). At low work this floor dominates and **masks true kernel
time** — latency does not drop below ~230 µs no matter how small the work. Therefore **only the
saturated points (tiles ≳ 512) reflect real kernel bandwidth**; the rest measure the harness.
This is a measurement artifact of the standalone driver, not the kernel (the HIP harness uses
the C++-bound op and has no such floor — hence the absolute-latency rows below differ even
where the kernels are equivalent).

### 3.2 Split-count sweep (tiles=256)

| splits | path | latency | achieved BW | % of 5.3 TB/s | note |
|---|---|---|---|---|---|
| 2 | simple | 235 µs | 717 GB/s | 14% | host-floor bound |
| 3 | simple | 440 µs | 535 GB/s | 10% | host-floor bound |
| 4 | massive | 445 µs | 680 GB/s | 13% | host-floor bound |
| 8 | massive | 282 µs | 2029 GB/s | 38% | partly floor-bound |
| 16 | massive | 453 µs | 2447 GB/s | 46% | |
| 32 | massive | 634 µs | 3448 GB/s | 65% | kernel-dominated |
| 64 | massive | 1247 µs | 3477 GB/s | 66% | kernel-dominated |
| 128 | massive | 2448 µs | 3529 GB/s | 67% | kernel-dominated |
| 256 | massive (tiles=64) | 1203 µs | 3585 GB/s | 68% | kernel-dominated |

The simple↔massive switch at splits=4 (`MASSIVE_THR`) is present. Below splits≈32 the per-call
host floor dominates the total traffic, so achieved-BW is understated; from splits≥32 the kernel
saturates at **3.45–3.59 TB/s (65–68%)**.

### 3.3 Batch (tile) sweep (splits=8)

| tiles | work items | latency | achieved BW | regime |
|---|---|---|---|---|
| 1 | 128 | 258 µs | 9 GB/s | host-floor |
| 4 | 512 | 241 µs | 37 GB/s | host-floor |
| 16 | 2048 | 449 µs | 80 GB/s | host-floor |
| 64 | 8192 | 240 µs | 595 GB/s | host-floor |
| 128 | 16384 | 234 µs | 1221 GB/s | host-floor |
| 256 | 32768 | 260 µs | 2202 GB/s | mixed |
| 512 | 65536 | 319 µs | 3584 GB/s | **kernel-dominated** |
| 1024 | 131072 | 619 µs | 3695 GB/s | kernel-dominated |
| 2048 | 262144 | 1246 µs | 3671 GB/s | kernel-dominated |
| 4096 | 524288 | 2499 µs | 3660 GB/s | kernel-dominated |

Once the kernel dominates (tiles ≥ 512), achieved BW is a stable **3.58–3.70 TB/s (68–70%)**.
The flat ~230 µs latency for tiles 1→128 is the host floor, not the kernel.

### 3.4 Other shapes / dtype (saturating batch)

| shape | dtype | splits | tiles | latency | achieved BW | % peak |
|---|---|---|---|---|---|---|
| H=128, Dv=512 | bf16 | 8 | 4096 | 2499 µs | 3660 GB/s | 69% |
| H=128, Dv=512 | fp16 | 8 | 4096 | 2485 µs | 3681 GB/s | 69% |
| H=16, Dv=512 | bf16 | 8 | 4096 | 335 µs | 3410 GB/s | 64% |
| H=128, Dv=128 | bf16 | 8 | 8192 | 1492 µs | 3083 GB/s | 58% |

bf16/fp16 are within noise (output dtype only changes the small final-O write). The smaller
`(128,128)` shape tops out a bit lower (58%) — less work per gather amortizes launch/index
overhead less well, consistent with its 4× smaller per-row payload.

---

## 4. Verdict

- **Correct** across the full shape × dtype × tier matrix (48/48), LSE exact (9.5e-7), output
  at bf16/fp16 rounding.
- **HBM-bandwidth-bound**, saturating at **~3.5–3.7 TB/s (65–70% of peak)** once kernel time
  dominates — squarely in the HIP kernel's measured band.
- The single benchmarking caveat is the standalone harness's ~230 µs host-call floor, which
  understates achieved-BW at low work. Always read the saturated rows for the kernel figure.

The kernel meets the prototype's stated bar — **HBM parity with the HIP reference, not a
speedup** (the reduction is at the byte floor; there is no compute headroom to win). See the
comparison report for the head-to-head.
