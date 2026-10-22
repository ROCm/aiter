# PR V4.1 fused FP8 attention optimization

The existing H128 FP8 QK path was slower than the complete PR #6675 V4.1 pipeline. This change fuses Q RoPE/group32 quantization and output BF16 rounding/inverse RoPE/group32 quantization into the V4.1 N64 attention body, while preserving the public max/LSE contract.

Baseline: `8512e0a4c6c2cca6fd4c214566f625284e2caf99`. PR reference: `b92f3cc9eeff2b5a5c56741a15da003dc5640d2e`, `mla_v41_prefill_a8w8_16mx4_64nx1.hpp`.

## Complete-path performance

MI355X/gfx950, GPU 6, unlocked clocks, original six fixture signatures, interleaved graph measurements. Times are microseconds. Each fused path includes max and LSE; PR includes Q preparation and output inverse RoPE/quantization, without LSE. Allocation, metadata construction, GEMM, cache writing and layout conversion are excluded.

| T / H / long S / K | Legacy before | Legacy after | Reduction | Paged before | Paged after | Reduction | PR complete |
|---|---:|---:|---:|---:|---:|---:|---:|
| 560 / 64 / 8192 / 640 | 95.05 | 95.04 | 0.02% | 94.89 | 94.95 | -0.06% | 113.56 |
| 560 / 64 / 32768 / 640 | 95.84 | 95.83 | 0.01% | 95.87 | 95.84 | 0.03% | 115.83 |
| 960 / 64 / 8192 / 640 | 126.59 | 126.92 | -0.26% | 125.23 | 124.81 | 0.34% | 179.75 |
| 960 / 64 / 32768 / 640 | 128.21 | 127.98 | 0.19% | 127.44 | 126.90 | 0.42% | 183.91 |
| 320 / 128 / 8192 / 1152 | 160.80 | 141.63 | 11.92% | 161.34 | 145.68 | 9.70% | 154.40 |
| 320 / 128 / 32768 / 1152 | 167.24 | 144.57 | 13.56% | 168.04 | 151.67 | 9.74% | 157.57 |

H64 keeps its previous kernels. Small changes there are measurement variation. On H128, native legacy and paged outputs match the direct selected code objects byte for byte. Raw-Q and prepared-Q outputs also match byte for byte.

## Selection and implementation

- Select only opt-in FP8 QK, H128, 1–320 queries, legacy tagged offsets or B128 paged cache, no BF16 extend, and prefix capacity at least `1024 * T`. H64, B7/B32/other page sizes, short lists and larger T retain the previous dispatch.
- Prefix capacity is a host-visible workload heuristic, not the actual valid count per query. Highly padded or skewed lists remain correct but may not benefit. The 12-case scan includes T16–2048 and K192–2176; it shows clear short-list losses and paged losses from T1024. An additional T512/S32768 probe showed marginal legacy gains and a small paged regression, so the final cutoff is T320. The final public-dispatch scan verifies selected and fallback cases.
- Each CTA validates the clamped CSR segment before choosing a template specialization of the attention loop. The common all-valid path avoids per-tile validity traffic. Invalid rows use a safe zero record and score masking. All-invalid tiles preserve zero output, max=-inf and LSE=+inf.
- Legacy addresses retain tag/alignment/bounds checks, including 4-byte-aligned records. Paged addressing uses 64-bit independent data/scale strides and host `backed_rows`, bounded by both byte views.
- FP32 exponential sums replace the original BF16-P MFMA sum. A separate true maximum preserves max_logits while the PR normalization maximum retains delayed rescaling. Existing strict tolerances are unchanged.
- Four code objects cover legacy/paged and raw/prepared Q. They share one source header and entry, and load through AiterAsmKernel. Q preparation and inverse RoPE use separate multiply/add instructions where required by the existing rounding contract.

## Validation and reproduction

- **277 passed**: all original 253 tests plus 24 new public API cases covering valid/mixed/truncated caches, raw/rotated/prepared Q, both position integer types, strict max/LSE and poisoned-output graph replay.
- 32 additional direct-contract cases include all-invalid queries and CSR clamping. Six complete-path fixtures and four public-dispatch scan fixtures check dense oracles and byte-exact replay.
- The consolidated source rebuild produces identical `.text` for all four measured kernels. No ordinary-ROCm source compilation of pinned-register attributes is assumed. Use `HIP_CLANG_PATH=/path/to/custom/llvm/bin python csrc/kernels/mla/fused_dsv41/pr_v41/rebuild_code_objects.py`; compiler commit is `583640bc544035fcc87177fee0410e0d87693919` from `https://github.com/yuyzhang512/llvm-project`, branch `amdgpu-named-reg-pin-std`.
- `results/` records raw samples, fixture signatures, replay checks, JUnit and code/source hashes. This is a kernel-path optimization; a complete model forward benchmark is outside the measured scope.

## Public-dispatch crossover checks

| T / S / K | Legacy before | Legacy after | Paged before | Paged after |
|---|---:|---:|---:|---:|
| 16 / 8192 / 1152 | 69.75 | 44.54 | 70.23 | 45.31 |
| 320 / 8192 / 256 | 53.51 | 53.50 | 53.73 | 53.61 |
| 1024 / 8192 / 1152 | 432.39 | 433.55 | 429.84 | 428.81 |
| 512 / 32768 / 1152 | 221.69 | 221.96 | 220.51 | 220.71 |

## Implementation diagram

[Legacy / paged addressing and fused pipeline](fused_dsv41_pr_v41_implementation.md) includes a source-linked SVG and an offline interactive HTML example.
