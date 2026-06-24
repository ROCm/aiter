# FlyDSL MLA decode-reduce — benchmark report

**Target:** AMD Instinct MI300X (gfx942, CDNA3, 304 CUs, warp 64), ROCm 7.2.0.
**Kernel:** `compile_mla_reduce` in `aiter/ops/flydsl/kernels/mla_reduce.py` — native FlyDSL
port of the HIP `kn_mla_reduce_v1`. Stage-2 epilogue of split-KV MLA decode: LSE-weighted
online-softmax combine of per-split partials. **Pure reduction — no MFMA, HBM-bandwidth bound.**

Companion to `MLA-reduce-HIP-kernel-benchmark-report.md` (the baseline) and
`MLA-reduce-flydsl-vs-hip-comparison.md` (the head-to-head). Reproduction steps:
`flydsl_mla_reduce_REPRODUCE.md`.

**Updated:** 2026-06-24 — numbers re-run after the multi-token decode fix (`decode_qlen > 1`,
empty-tile guard). Partial inputs model **stage-1 fp32 outputs** (see §1).

---

## 1. Environment

| Component | Version / value |
|---|---|
| GPU | AMD Instinct MI300X (gfx942, 304 CUs, warp 64), HBM3 ~5.3 TB/s peak |
| Container | `rocm/ali-private:ubuntu22.04_rocm7.2.0.43_cp310_torch2.9.1_sglang_027c46b_aiter_eee0abe_qwen3_5_20260529` |
| ROCm / Python / PyTorch | 7.2.0 / 3.10.12 / 2.9.1+rocm7.2.0 |
| flydsl | 0.2.0 |

**Input model (stage-1 partials).** The reduce kernel always reads **fp32** partial `O` and
`LSE` buffers — the same dtype/layout the MLA stage-1 ASM kernel writes. Upstream Q/KV may be
bf16 or fp8; stage-1 quantizes/dequantizes internally and still emits fp32 partials, so reduce
bandwidth is identical for bf16 vs fp8 upstream (only the small final-`O` write scales with
output dtype). The standalone harness (`build_inputs`) fills fp32 partials in the
`get_mla_metadata_v1` CSR layout; end-to-end paths (`test_mla_sparse.py`, FlyDSL on) exercise
real stage-1 → reduce sequencing including **fp8 qlen=2** (72/72 clean via
`stress_flydsl_mla_reduce.sh`).

Harness: `op_tests/test_flydsl_mla_reduce.py --bench`, 25 warmup / 100 timed iters.
Latency is **kernel-only device time** from aiter's `run_perftest`
(`aiter/test_common.py`, `torch.profiler` → `self_device_time_total`, IQR-filtered),
with a CUDA-graph `cuda.Event` replay as an independent cross-check (`graph` column).
Both exclude per-call Python host overhead. BW = traffic model (`T·S·H·Dv·4`
partial-O read + lse + final-O write) / kernel time.

---

## 2. Correctness

Full matrix — 3 shapes `(H,Dv) ∈ {(128,512),(16,512),(128,128)}` × {bf16, fp16} × 8 split
counts `{2,3,4,8,16,64,256,300}` × `M = max_seqlen_q ∈ {1,2,4}` plus 12 degenerate empty-tile
guard cases (`--degenerate`, also folded into `--matrix`):

**156 / 156 pass.** The reference is the **HIP `kn_mla_reduce_v1` kernel itself** — run on
the identical input buffers. The `M > 1` rows mirror the stage-1 row layout (multiple partial
rows per split); production multi-token decode (`decode_qlen > 1`) uses **`M = 1` with more
reduce tiles** and is covered by the e2e stress script, not by widening `M` in metadata.
Degenerate rows (`reduce_indptr = [0,0,…]`, garbage `reduce_final_map`) lock in the
empty-tile guard added for qlen>1.

Output `max_abs_err ≤ 3.12e-2` (bf16) / `1.95e-3` (fp16); LSE `max_abs_err ≤ 9.5e-7`.
The bf16 output figure is **exactly one bf16 ULP** (2⁻⁵ ≈ 3.1e-2): both kernels compute
in fp32 and round the final result to bf16 independently.

---

## 3. Benchmark results

Shape **H=128, Dv=512, bf16** (GLM/DeepSeek decode shape) unless noted. Partials are fp32
(stage-1 layout); sweeps use synthetic fp32 fills (byte traffic identical to real stage-1 output).

### 3.1 Split-count sweep (tiles=256)

| splits | path | kernel | CUDA graph | achieved BW | % of 5.3 TB/s |
|---|---|---|---|---|---|
| 2 | simple | 36.7 µs | 35.6 µs | 4577 GB/s | 86% |
| 3 | simple | 50.7 µs | 50.3 µs | 4647 GB/s | 88% |
| 4 | massive | 78.0 µs | 82.5 µs | 3882 GB/s | 73% |
| 8 | massive | 161.9 µs | 167.6 µs | 3530 GB/s | 67% |
| 16 | massive | 404.4 µs | 378.0 µs | 2744 GB/s | 52% |
| 32 | massive | 700.8 µs | 702.4 µs | 3118 GB/s | 59% |
| 64 | massive | 1267.4 µs | 1273.4 µs | 3422 GB/s | 65% |
| 128 | massive | 2405.4 µs | 2390.9 µs | 3592 GB/s | 68% |
| 256 | massive (tiles=64) | 1164.3 µs | 1166.1 µs | 3703 GB/s | 70% |

The simple↔massive switch at splits=4 (`MASSIVE_THR`) is visible as a BW step. The **simple
path (splits 2–3) runs at 86–88%**. The massive path settles into **3.4–3.7 TB/s (64–70%)**
once split count is high enough to amortize pipeline overhead.

### 3.2 Batch (tile) sweep (splits=8)

| tiles | work items | kernel | CUDA graph | achieved BW | regime |
|---|---|---|---|---|---|
| 1 | 128 | 5.4 µs | 3.9 µs | 415 GB/s | latency-bound |
| 4 | 512 | 5.5 µs | 4.0 µs | 1633 GB/s | latency-bound |
| 16 | 2048 | 8.1 µs | 6.5 µs | 4393 GB/s | near peak |
| 64 | 8192 | 34.9 µs | 33.3 µs | 4092 GB/s | saturating |
| 128 | 16384 | 65.6 µs | 63.7 µs | 4354 GB/s | saturating |
| 256 | 32768 | 164.5 µs | 167.5 µs | 3475 GB/s | kernel-dominated |
| 512 | 65536 | 335.9 µs | 333.2 µs | 3404 GB/s | kernel-dominated |
| 1024 | 131072 | 670.6 µs | 663.2 µs | 3409 GB/s | kernel-dominated |
| 2048 | 262144 | 1338.9 µs | 1335.2 µs | 3415 GB/s | kernel-dominated |
| 4096 | 524288 | 2688.3 µs | 2690.3 µs | 3402 GB/s | kernel-dominated |

Latency-bound only at tiles 1–4. By tiles≈16 the kernel touches peak (~4.4 TB/s, small-footprint
cache effect), then settles to a stable **3.4 TB/s (64%)** in the streaming regime.

### 3.3 Other shapes / dtype (saturating batch)

| shape | dtype | splits | tiles | kernel | CUDA graph | achieved BW | % peak |
|---|---|---|---|---|---|---|---|
| H=128, Dv=512 | bf16 | 8 | 4096 | 2688 µs | 2690 µs | 3402 GB/s | 64% |
| H=128, Dv=512 | fp16 | 8 | 4096 | 2695 µs | 2698 µs | 3393 GB/s | 64% |
| H=16, Dv=512 | bf16 | 8 | 4096 | 325 µs | 326 µs | 3513 GB/s | 66% |
| H=128, Dv=128 | bf16 | 8 | 8192 | 1470 µs | 1478 µs | 3130 GB/s | 59% |

bf16/fp16 differ only in the small final-O write (partials remain fp32 stage-1 layout).
Smaller `(128,128)` tops out lower (59%) — less work per gather row.

### 3.4 Production-path notes (stage-1 → reduce, fp8)

End-to-end sparse MLA decode (`test_mla_sparse.py`, `-n16,{1,2} -b1 -c21 -k512`) runs
stage-1 ASM → fp32 partials → reduce. With `AITER_MLA_REDUCE_FLYDSL=1`:

| config | e2e result | reduce-only BW implication |
|---|---|---|
| bf16 qlen=1/2 | clean, `decode:err = 0` | same fp32 partial traffic as §3 |
| fp8 qlen=1/2 | clean, fp8 quant delta only (~0.36 err vs bf16 golden) | partials still fp32 → same BW band |

Reduce-only device time is not separately reported in the e2e harness (stage-1 dominates total
latency at this shape). Standalone §3 numbers are the authoritative reduce kernel BW.

---

## 4. Verdict

- **Correct** across the full shape × dtype × tier × M matrix plus degenerate guard (156/156),
  validated directly against the HIP kernel.
- **HBM-bandwidth-bound**, saturating at **~3.4–3.7 TB/s (64–70%)** on the massive path in
  the streaming regime — in the HIP kernel's measured band; simple path reaches 86–88%.
- Inputs model **stage-1 fp32 partials**; bf16 and fp8 upstream paths share the same reduce
  byte floor. Multi-token decode (`decode_qlen > 1`) is e2e-validated (bf16/fp8, 72/72 stress).

The kernel meets the prototype's stated bar — **HBM parity with the HIP reference**. See the
comparison report for the head-to-head.
