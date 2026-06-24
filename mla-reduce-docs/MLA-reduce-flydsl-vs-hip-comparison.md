# MLA decode-reduce — FlyDSL vs HIP comparison report

Head-to-head between the **FlyDSL prototype** (`compile_mla_reduce`,
`aiter/ops/flydsl/kernels/mla_reduce.py`) and the **HIP reference** (`kn_mla_reduce_v1` /
`_ps`, `aiter/csrc/kernels/mla/reduce.cu`), both measured on the same MI300X / ROCm 7.2.0
container. Sources:

- HIP baseline: `MLA-reduce-HIP-kernel-benchmark-report.md`
- FlyDSL: `MLA-reduce-flydsl-benchmark-report.md`
- Design: `MLA-reduce-HIP-kernel-dissection.md`, `MLA-reduce-flydsl-moe-reduction-dissection.md`
- Multi-token fix: `MLA-reduce-flydsl-multitoken-decode-fix-report.md`

**Updated:** 2026-06-24 — re-run after multi-token decode fix; partial inputs model stage-1
fp32 outputs (bf16/fp8 upstream).

**Bottom line:** the FlyDSL port reaches the same HBM-bandwidth band as HIP
(**~3.2–3.7 TB/s, 60–70% of peak**) with bit-comparable accuracy. Both are at the reduction
byte floor. Both sides use **kernel-only device time** (aiter's `run_perftest` + CUDA-graph
cross-check). End-to-end stage-1 → reduce paths (bf16/fp8, qlen 1–4) are stress-validated
with FlyDSL on (72/72 clean).

---

## 1. Environment (identical for both)

| | value |
|---|---|
| GPU | AMD Instinct MI300X (gfx942, 304 CUs, warp 64), HBM3 ~5.3 TB/s peak |
| Container | `rocm/ali-private:...rocm7.2.0...torch2.9.1...` |
| ROCm / Python / PyTorch | 7.2.0 / 3.10.12 / 2.9.1+rocm7.2.0 |
| flydsl | 0.2.0 |
| Timing | 25 warmup / 100 iters, **kernel-only device time** (`run_perftest` `self_device_time_total` + CUDA-graph `cuda.Event` cross-check); same traffic model |
| Inputs | fp32 partial `O`/`LSE` in stage-1 output layout (synthetic fill in standalone sweeps; real stage-1 ASM in e2e) |

---

## 2. Correctness

The FlyDSL matrix is checked **directly against the HIP `kn_mla_reduce_v1` kernel** (same
input buffers, kernel-vs-kernel). **156/156 configs pass** (144 well-formed × `M∈{1,2,4}` +
12 degenerate empty-tile guard cases for qlen>1).

| metric | FlyDSL vs HIP |
|---|---|
| output max_abs_err (bf16) | ≤ 3.12e-2 — **one bf16 ULP** (2⁻⁵) |
| output max_abs_err (fp16) | ≤ 1.95e-3 |
| LSE max_abs_err | ≤ 9.5e-7 |
| supported `(H,Dv)` | (128,512),(16,512),(128,128) — same three |
| e2e stage-1 input (FlyDSL on) | bf16/fp8 qlen 1–4, 72/72 clean (`stress_flydsl_mla_reduce.sh`) |
| reference | HIP `kn_mla_reduce_v1` (torch fp64 ref also available) |

**LSE (fp32) matches to ~1e-7.** The bf16 output diff is exactly one bf16 ULP. Production
multi-token decode uses **`M = 1` with more reduce tiles** (not wider `max_seqlen_q`); the
empty-tile guard skips degenerate `n_splits = 0` tiles that stage-1/metadata can emit.

---

## 3. Bandwidth — split-count sweep (H=128, Dv=512, bf16)

HIP at tiles=256; FlyDSL at tiles=256 (splits=256 row at tiles=64 for both).
Partials: fp32 stage-1 layout.

| splits | path | HIP kernel | HIP BW | FlyDSL kernel | FlyDSL BW | notes |
|---|---|---|---|---|---|---|
| 2 | simple | 49.4 µs | 3.40 TB/s | 36.7 µs | **4.58 TB/s** | FlyDSL faster on simple path |
| 3 | simple | 66.3 µs | 3.55 TB/s | 50.7 µs | **4.65 TB/s** | FlyDSL faster |
| 4 | massive | 78.8 µs | 3.84 TB/s | 78.0 µs | 3.88 TB/s | ~parity |
| 8 | massive | 177.1 µs | 3.23 TB/s | 161.9 µs | 3.53 TB/s | FlyDSL slightly ahead |
| 16 | massive | 350.8 µs | 3.16 TB/s | 404.4 µs | 2.74 TB/s | FlyDSL slower (mid-split pipeline) |
| 32 | massive | 668.0 µs | 3.27 TB/s | 700.8 µs | 3.12 TB/s | ~parity |
| 64 | massive | 1282.0 µs | 3.38 TB/s | 1267.4 µs | 3.42 TB/s | ~parity |
| 128 | massive | 2519.2 µs | 3.43 TB/s | 2405.4 µs | 3.59 TB/s | ~parity |
| 256 | massive (t=64) | 1195.2 µs | 3.61 TB/s | 1164.3 µs | 3.70 TB/s | ~parity |

Both track within a few percent across most of the sweep. FlyDSL remains faster on the **simple
path (splits 2–3)** and at high split counts; mid-split (splits=16) shows a FlyDSL pipeline
overhead bump that is worth deeper prefetch tuning. fp8 upstream (stage-1 input) does not change
reduce traffic — partials stay fp32.

---

## 4. Bandwidth — batch sweep (H=128, Dv=512, bf16, splits=8)

| tiles | HIP kernel | HIP BW | FlyDSL kernel | FlyDSL BW | regime |
|---|---|---|---|---|---|
| 1 | 4.4 µs | 504 GB/s | 5.4 µs | 415 GB/s | latency-bound |
| 4 | 4.6 µs | 1.94 TB/s | 5.5 µs | 1.63 TB/s | latency-bound |
| 16 | 7.8 µs | **4.58 TB/s** | 8.1 µs | **4.39 TB/s** | near peak (cache) |
| 64 | 46.1 µs | 3.10 TB/s | 34.9 µs | 4.09 TB/s | saturating |
| 128 | 73.5 µs | 3.89 TB/s | 65.6 µs | 4.35 TB/s | saturating |
| 256 | 174.4 µs | 3.28 TB/s | 164.5 µs | 3.48 TB/s | kernel-dominated |
| 512 | 341.9 µs | 3.34 TB/s | 335.9 µs | 3.40 TB/s | kernel-dominated |
| 1024 | 687.2 µs | 3.33 TB/s | 670.6 µs | 3.41 TB/s | kernel-dominated |
| 2048 | — | — | 1338.9 µs | 3.42 TB/s | kernel-dominated |
| 4096 | — | — | 2688.3 µs | 3.40 TB/s | kernel-dominated |

Both are latency-bound only at tiles 1–4 and touch peak near tiles≈16. In the streaming
regime (tiles ≥ 256) both settle at **~3.3–3.5 TB/s** — parity at the byte floor.

---

## 5. Design / structure parity

| dimension | HIP `kn_mla_reduce_v1` | FlyDSL `compile_mla_reduce` |
|---|---|---|
| math | LSE-weighted online softmax | same |
| block | 128 threads / 2 waves | same (`known_block_size=[128,1,1]`) |
| output partition | `Dv/128` floats/thread (lane) | same (`VEC=Dv//128`) |
| simple path (<4 splits) | register online-softmax | runtime `range(init=)` carried scalars |
| massive path (≥4) | warp-0 LSE + LDS scale + 2-way pipeline | warp-0 shuffle LSE + LDS `lse_scale` + barrier |
| tiers | runtime template (≤64/≤256/>256 spill) | compile-time tier (m64/m256/mlds), host-selected |
| simple↔massive | runtime | runtime `scf.IfOp` (within one compiled tier) |
| empty tile (n_splits≤1) | skip guards (`reduce.cu:691,743`) | `has_work`/`ub_seq` clamp (qlen>1 fix) |
| LDS | gather map + lse_scale (no O data) | same; tiny by design (preserve occ-8) |
| launch | grid-launch + persistent grid-stride | grid-launch (persistent deferred) |
| exp / log | `exp2`/native | `rocdl.exp2(x·log2e)` / `mlir_math.log` |
| stage-1 contract | fp32 partial `O`/`LSE` gather | same pointers/layout |

**Functional structure is a faithful 1:1 port.** FlyDSL selects the split-tier at compile time;
persistent grid-stride launcher deferred (parity-neutral at the byte floor).

---

## 6. Verdict

| question | answer |
|---|---|
| Correct vs HIP? | **Yes** — 156/156 standalone + 72/72 e2e (bf16/fp8 qlen 1–4, real stage-1 input) |
| HBM parity? | **Yes** — ~3.3–3.7 TB/s streaming, equal-to-slightly-ahead except mid-split bump |
| Faster? | At parity — simple path and high splits slightly favor FlyDSL; both at byte floor |
| fp8 stage-1 input? | **Same reduce BW** — stage-1 emits fp32 partials; fp8 delta is upstream quant only |
| Caveats | Mid-split (splits≈16) FlyDSL pipeline overhead; persistent launcher deferred. Production wiring opt-in: `AITER_MLA_REDUCE_FLYDSL=1` |

**The FlyDSL prototype achieves its goal: a native, correct, HBM-parity replacement for the
HIP MLA decode-reduce on gfx942, including multi-token decode with real stage-1 partial inputs.**
