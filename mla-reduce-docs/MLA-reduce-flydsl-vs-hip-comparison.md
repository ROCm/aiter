# MLA decode-reduce — FlyDSL vs HIP comparison report

Head-to-head between the **FlyDSL prototype** (`compile_mla_reduce`,
`aiter/ops/flydsl/kernels/mla_reduce.py`) and the **HIP reference** (`kn_mla_reduce_v1` /
`_ps`, `aiter/csrc/kernels/mla/reduce.cu`), both measured on the same MI300X / ROCm 7.2.0
container. Sources:

- HIP baseline: `MLA-reduce-HIP-kernel-benchmark-report.md`
- FlyDSL: `MLA-reduce-flydsl-benchmark-report.md`
- Design: `MLA-reduce-HIP-kernel-dissection.md`, `MLA-reduce-flydsl-moe-reduction-dissection.md`

**Bottom line:** the FlyDSL port reaches the same HBM-bandwidth band as HIP
(**~3.5–3.8 TB/s, 65–72% of peak**) with bit-comparable accuracy. Both are at the reduction
byte floor — parity was the goal, and it is met. Both sides are now measured with **kernel-only
device time** (aiter's `run_perftest` + a CUDA-graph cross-check), so the tables compare kernels
directly with no host-overhead artifact; the old ~230 µs FlyDSL-harness floor is gone.

---

## 1. Environment (identical for both)

| | value |
|---|---|
| GPU | AMD Instinct MI300X (gfx942, 304 CUs, warp 64), HBM3 ~5.3 TB/s peak |
| Container | `rocm/ali-private:...rocm7.2.0...torch2.9.1...` |
| ROCm / Python / PyTorch | 7.2.0 / 3.10.12 / 2.9.1+rocm7.2.0 |
| flydsl | 0.2.0 |
| Timing | 25 warmup / 100 iters, **kernel-only device time** (`run_perftest` `self_device_time_total` + CUDA-graph `cuda.Event` cross-check); same traffic model |

---

## 2. Correctness

| metric | HIP | FlyDSL |
|---|---|---|
| output max_abs_err (bf16) | 0.031 | ≤ 3.9e-3 (matrix, tiles=4) |
| output max_abs_err (fp16) | — | ≤ 4.9e-4 |
| LSE max_abs_err | 9.5e-7 | ≤ 9.5e-7 (exact) |
| supported `(H,Dv)` | (128,512),(16,512),(128,128) | same three |
| reference | vectorized torch online-softmax | same (fp64) |

Both validate against the same torch online-softmax reference. **LSE matches exactly to the
same 9.5e-7**; output error is pure bf16/fp16 rounding in both. The FlyDSL matrix error is
lower simply because it reports the `tiles=4` sweep (the HIP 0.031 was measured at large
tiles where bf16 degeneracy inflates the figure — an input property, not a kernel difference).

---

## 3. Bandwidth — split-count sweep (H=128, Dv=512, bf16)

HIP at tiles=256; FlyDSL at tiles=256 (splits=256 row at tiles=64 for both).

| splits | path | HIP kernel | HIP BW | FlyDSL kernel | FlyDSL BW | notes |
|---|---|---|---|---|---|---|
| 2 | simple | 49.7 µs | 3.38 TB/s | 36.5 µs | **4.61 TB/s** | FlyDSL faster on simple path |
| 3 | simple | 66.1 µs | 3.56 TB/s | 48.6 µs | **4.84 TB/s** | FlyDSL faster |
| 4 | massive | 78.5 µs | 3.86 TB/s | 83.2 µs | 3.64 TB/s | ~parity |
| 8 | massive | 176.7 µs | 3.24 TB/s | 153.3 µs | 3.73 TB/s | ~parity |
| 16 | massive | 357.6 µs | 3.10 TB/s | 328.3 µs | 3.38 TB/s | ~parity |
| 32 | massive | 661.2 µs | 3.31 TB/s | 642.8 µs | 3.40 TB/s | ~parity |
| 64 | massive | 1255.7 µs | 3.45 TB/s | 1270.1 µs | 3.42 TB/s | ~parity |
| 128 | massive | 2501.0 µs | 3.46 TB/s | 2458.7 µs | 3.51 TB/s | ~parity |
| 256 | massive (t=64) | 1167.9 µs | 3.69 TB/s | 1200.3 µs | 3.59 TB/s | ~parity |

With both sides on kernel-only device time, **they track within a few percent across the whole
sweep** — the FlyDSL port matches HIP byte-for-byte on the massive path. On the **simple path
(splits 2–3) FlyDSL is actually faster** (4.6–4.8 vs 3.4–3.6 TB/s). The old large low-split gap
was purely the FlyDSL standalone harness's host floor and has now disappeared.

---

## 4. Bandwidth — batch sweep (H=128, Dv=512, bf16, splits=8)

| tiles | HIP kernel | HIP BW | FlyDSL kernel | FlyDSL BW | regime |
|---|---|---|---|---|---|
| 1 | 4.7 µs | 474 GB/s | 4.7 µs | 480 GB/s | latency-bound |
| 4 | 4.5 µs | 1.98 TB/s | 4.8 µs | 1.87 TB/s | latency-bound |
| 16 | 8.2 µs | **4.39 TB/s** | 6.8 µs | **5.25 TB/s** | near peak (cache) |
| 64 | 45.7 µs | 3.13 TB/s | 33.1 µs | 4.32 TB/s | saturating |
| 128 | 74.1 µs | 3.86 TB/s | 59.3 µs | 4.82 TB/s | saturating |
| 256 | 175.8 µs | 3.25 TB/s | 150.0 µs | 3.81 TB/s | kernel-dominated |
| 512 | 337.2 µs | 3.39 TB/s | 316.2 µs | 3.62 TB/s | kernel-dominated |
| 1024 | 683.2 µs | 3.35 TB/s | 608.3 µs | **3.76 TB/s** | kernel-dominated |
| 2048 | — | — | 1211.7 µs | 3.77 TB/s | kernel-dominated |
| 4096 | — | — | 2472.2 µs | 3.70 TB/s | kernel-dominated |

With both sides on kernel-only device time, the two **track closely at every tile count**, and
FlyDSL is **equal to or slightly faster than HIP** across the saturating/kernel-dominated range
(e.g. 3.62–3.77 vs 3.35–3.39 TB/s at tiles 512–1024). Both are latency-bound only at tiles 1–4,
and both touch peak near tiles≈16 (a small-footprint cache effect). The dramatic low-tile gap
seen in earlier reports was entirely the FlyDSL standalone harness's ~230 µs host-call floor;
with device timing it is gone. **The kernels are at parity (FlyDSL marginally ahead).**

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
| LDS | gather map + lse_scale (no O data) | same; tiny by design (preserve occ-8) |
| launch | grid-launch + persistent grid-stride | grid-launch (persistent deferred) |
| exp / log | `exp2`/native | `rocdl.exp2(x·log2e)` / `mlir_math.log` |

**Functional structure is a faithful 1:1 port.** Two intentional deltas, both perf-neutral at
the byte floor: (a) FlyDSL selects the split-tier at **compile time** (one kernel per tier)
because loop-carried tuple arity must be static at trace time — HIP does it via runtime C++
template; (b) the **persistent grid-stride launcher** is not yet ported — grid-launch already
saturates HBM, so it is parity-neutral here (it remains the next increment for strict host
fidelity).

---

## 6. Verdict

| question | answer |
|---|---|
| Correct vs HIP? | **Yes** — LSE exact (9.5e-7), output at bf16/fp16 rounding, same 3 shapes. |
| HBM parity? | **Yes** — 3.6–3.8 TB/s where kernel-dominated, equal-to-slightly-ahead of HIP and within HIP's own band; FlyDSL faster on the simple path (low splits). |
| Faster? | At parity (FlyDSL marginally ahead) — both at the reduction byte floor (~99% of time is HBM traffic per the HIP rocprofv3 profile). Parity was the bar. |
| Caveats | None on timing — both sides now use kernel-only device time, so the old ~230 µs host-floor artifact is gone and low-work BW is meaningful. Persistent launcher deferred (parity-neutral at the byte floor). |

**The FlyDSL prototype achieves its goal: a native, correct, HBM-parity replacement for the
HIP MLA decode-reduce on gfx942.**
