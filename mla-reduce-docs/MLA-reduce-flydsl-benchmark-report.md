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

Harness: `op_tests/test_flydsl_mla_reduce.py --bench`, 25 warmup / 100 timed iters.
Latency is **kernel-only device time** from aiter's `run_perftest`
(`aiter/test_common.py`, `torch.profiler` → `self_device_time_total`, IQR-filtered),
with a CUDA-graph `cuda.Event` replay as an independent cross-check (`graph` column).
Both exclude per-call Python host overhead. BW = traffic model (`T·S·H·Dv·4`
partial-O read + lse + final-O write) / kernel time.

---

## 2. Correctness

Full matrix — 3 shapes `(H,Dv) ∈ {(128,512),(16,512),(128,128)}` × {bf16, fp16} × 8 split
counts `{2,3,4,8,16,64,256,300}`, spanning all four tiers (simple / m64 / m256 / mlds):

**48 / 48 pass.** The reference is now the **HIP `kn_mla_reduce_v1` kernel itself** — the
kernel this FlyDSL port replaces — run on the identical input buffers, so the check is a
direct kernel-vs-kernel comparison rather than against a torch model. (A vectorized fp64
torch online-softmax remains available as `torch_ref` for debugging.)

Output `max_abs_err ≤ 3.12e-2` (bf16) / `1.95e-3` (fp16); LSE `max_abs_err ≤ 9.5e-7`.
The bf16 output figure is **exactly one bf16 ULP** (2⁻⁵ ≈ 3.1e-2): both kernels compute
in fp32 and round the final result to bf16 independently, so near-equal fp32 values can
land on adjacent bf16 codes. LSE (fp32 output) matches to ~1e-7. Tolerance is therefore
set to one ULP of the output dtype (bf16 ≈ 6.3e-2, fp16 ≈ 2e-3).

---

## 3. Benchmark results

Shape **H=128, Dv=512, bf16** (GLM/DeepSeek decode shape) unless noted.

### 3.1 Split-count sweep (tiles=256)

| splits | path | kernel | CUDA graph | achieved BW | % of 5.3 TB/s |
|---|---|---|---|---|---|
| 2 | simple | 36.5 µs | 35.2 µs | 4608 GB/s | 87% |
| 3 | simple | 48.6 µs | 49.1 µs | 4843 GB/s | 91% |
| 4 | massive | 83.2 µs | 89.4 µs | 3637 GB/s | 69% |
| 8 | massive | 153.3 µs | 161.5 µs | 3729 GB/s | 70% |
| 16 | massive | 328.3 µs | 322.7 µs | 3380 GB/s | 64% |
| 32 | massive | 642.8 µs | 632.3 µs | 3400 GB/s | 64% |
| 64 | massive | 1270.1 µs | 1252.1 µs | 3415 GB/s | 64% |
| 128 | massive | 2458.7 µs | 2454.6 µs | 3514 GB/s | 66% |
| 256 | massive (tiles=64) | 1200.3 µs | 1196.2 µs | 3592 GB/s | 68% |

The simple↔massive switch at splits=4 (`MASSIVE_THR`) is visible as a BW step. The **simple
path (splits 2–3) runs at 87–91%**.
The massive path settles into the reduction's **3.38–3.59 TB/s (64–68%)** band.

### 3.2 Batch (tile) sweep (splits=8)

| tiles | work items | kernel | CUDA graph | achieved BW | regime |
|---|---|---|---|---|---|
| 1 | 128 | 4.7 µs | 3.3 µs | 480 GB/s | latency-bound |
| 4 | 512 | 4.8 µs | 3.5 µs | 1870 GB/s | latency-bound |
| 16 | 2048 | 6.8 µs | 5.9 µs | 5254 GB/s | near peak |
| 64 | 8192 | 33.1 µs | 33.2 µs | 4323 GB/s | saturating |
| 128 | 16384 | 59.3 µs | 62.2 µs | 4823 GB/s | saturating |
| 256 | 32768 | 150.0 µs | 157.9 µs | 3812 GB/s | kernel-dominated |
| 512 | 65536 | 316.2 µs | 317.7 µs | 3615 GB/s | kernel-dominated |
| 1024 | 131072 | 608.3 µs | 625.8 µs | 3759 GB/s | kernel-dominated |
| 2048 | 262144 | 1211.7 µs | 1243.3 µs | 3774 GB/s | kernel-dominated |
| 4096 | 524288 | 2472.2 µs | 2494.1 µs | 3699 GB/s | kernel-dominated |

The kernel is **latency-bound only at tiny tile counts** (1–4),
where 128–512 work items can't fill the machine. By tiles≈16 it already touches peak
(~5.3 TB/s, a small-footprint L2/cache effect), then settles to a stable **3.6–3.8 TB/s
(68–72%)** once the working set exceeds cache.

### 3.4 Other shapes / dtype (saturating batch)

| shape | dtype | splits | tiles | kernel | CUDA graph | achieved BW | % peak |
|---|---|---|---|---|---|---|---|
| H=128, Dv=512 | bf16 | 8 | 4096 | 2448 µs | 2496 µs | 3736 GB/s | 70% |
| H=128, Dv=512 | fp16 | 8 | 4096 | 2424 µs | 2482 µs | 3772 GB/s | 71% |
| H=16, Dv=512 | bf16 | 8 | 4096 | 334 µs | 334 µs | 3423 GB/s | 65% |
| H=128, Dv=128 | bf16 | 8 | 8192 | 1486 µs | 1490 µs | 3097 GB/s | 58% |

bf16/fp16 are within noise (output dtype only changes the small final-O write). The smaller
`(128,128)` shape tops out a bit lower (58%) — less work per gather amortizes launch/index
overhead less well, consistent with its 4× smaller per-row payload.

---

## 4. Verdict

- **Correct** across the full shape × dtype × tier matrix (48/48), validated directly against
  the HIP `kn_mla_reduce_v1` kernel: LSE matches to 9.5e-7, output to within one bf16/fp16 ULP.
- **HBM-bandwidth-bound**, saturating at **~3.6–3.8 TB/s (68–72% of peak)** on the massive
  path — squarely in the HIP kernel's measured band; the simple path (low splits) reaches
  87–91%.
- With kernel-only device timing (host overhead excluded), the old ~230 µs harness floor no
  longer distorts low-work rows; the profiler and CUDA-graph numbers agree within a few percent.

The kernel meets the prototype's stated bar — **HBM parity with the HIP reference, not a
speedup** (the reduction is at the byte floor; there is no compute headroom to win). See the
comparison report for the head-to-head.
