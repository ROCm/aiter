# gfx1250 C++ RMSNorm wave32 dispatch

## Changes and scope

Parent: `6343508e8d4c4d550b439068034029bde94b4e2d`.

MI455 (gfx1250) uses wave32, not gfx950's wave64. This change does not
alter global `WARP_SIZE` or apply gfx1250 dispatch to other devices.

* `block_reduce`: blocks <= one hardware wave use a logical `BlockSize`
  wave reduction. A 32-thread instantiation also compiles in the HIP host
  pass where the constexpr warp-size fallback is 64. Existing full-wave
  and larger-block reductions keep the same behavior.
* Single-input **no-quant** fused QK/Gemma RMSNorm:
  * N=128: 32 threads, four values/thread, no cross-wave reduction.
  * N=6144, M>=256: 256 threads, 24 values/thread, interleaved wave-sized
    vector accesses. The old 512x16 block has only 384 valid threads here.
  * M<256 at N=6144 keeps the old dispatch; an all-M candidate regressed
    M=1/2 and was rejected.
* Generic no-quant rmsnorm/add_rmsnorm: N=128 uses 32x4 on gfx1250.
* Quantized output, dual Q/K input, other widths and other architectures
  keep the existing dispatch. FP32 residual addition and Gemma `(1+w)`
  semantics are unchanged.

A 256x24 candidate without interleave regressed; it is not included.

## Same-machine measurement

2026-10-28 UTC; one gfx1250 GPU (256 CU), no clock/power changes.
PyTorch `2.11.0+rocm10.1.0a20260811-bkc.20260820`, HIP `7.16.26315`, driver `7.1.0.31300009`.
Before/after use the same source parent, toolchain, seed and GPU, with
separate source checkouts and JIT caches.

BF16, eps=1e-6, Gemma enabled; N=6144 norm/add+norm and N=128 norm.
198 cases/pass, process order baseline / optimized / optimized / baseline.
Seed=2026 and the same shuffled shape order in every pass. HIP Graph
captures 128 calls; 20 warmup calls and four graph warmups; nine samples
>=5ms per case/pass. Tables use pooled medians of 18 samples per revision.
Buffers are preallocated/reused (warm cache); CPU reference, allocation
and JIT are outside timing.

All 792 measured case-runs passed rtol=0.01/atol=0.01 against FP32
reference before/after timing; residual outputs matched exactly.
The FP32-reference error includes BF16 output rounding.

This is not an MI355-versus-MI455 comparison or end-to-end model speedup.
No allreduce, quantization, RoPE or cache insertion is timed.

### Representative results

| N | M | Path | Mode | Before (us) | After (us) | Before / after |
|---:|---:|---|---|---:|---:|---:|
| 128 | 1 | atom | norm | 2.046 | 2.187 | 0.935x |
| 128 | 1 | generic | norm | 1.676 | 1.574 | 1.065x |
| 128 | 128 | atom | norm | 2.587 | 2.718 | 0.952x |
| 128 | 128 | generic | norm | 1.815 | 1.706 | 1.064x |
| 128 | 16384 | atom | norm | 5.625 | 4.682 | 1.202x |
| 128 | 16384 | generic | norm | 5.528 | 4.644 | 1.190x |
| 128 | 131072 | atom | norm | 22.926 | 19.696 | 1.164x |
| 128 | 131072 | generic | norm | 26.728 | 19.707 | 1.356x |
| 128 | 524288 | atom | norm | 81.823 | 71.315 | 1.147x |
| 128 | 524288 | generic | norm | 99.169 | 71.339 | 1.390x |
| 6144 | 1 | atom | add_norm | 2.250 | 2.264 | 0.994x |
| 6144 | 1 | atom | norm | 2.758 | 2.605 | 1.059x |
| 6144 | 2 | atom | add_norm | 2.462 | 2.623 | 0.939x |
| 6144 | 2 | atom | norm | 2.581 | 2.583 | 0.999x |
| 6144 | 128 | atom | add_norm | 3.809 | 3.838 | 0.992x |
| 6144 | 128 | atom | norm | 3.490 | 3.261 | 1.070x |
| 6144 | 256 | atom | add_norm | 4.031 | 4.047 | 0.996x |
| 6144 | 256 | atom | norm | 4.036 | 3.998 | 1.009x |
| 6144 | 1024 | atom | add_norm | 7.025 | 5.865 | 1.198x |
| 6144 | 1024 | atom | norm | 4.718 | 4.417 | 1.068x |
| 6144 | 4096 | atom | add_norm | 20.584 | 16.288 | 1.264x |
| 6144 | 4096 | atom | norm | 13.026 | 9.937 | 1.311x |
| 6144 | 16384 | atom | add_norm | 73.351 | 57.452 | 1.277x |
| 6144 | 16384 | atom | norm | 41.577 | 30.371 | 1.369x |
| 6144 | 32768 | atom | add_norm | 155.875 | 121.067 | 1.288x |
| 6144 | 32768 | atom | norm | 82.586 | 57.565 | 1.435x |

Full unfiltered results: [rmsnorm_gfx1250.csv](rmsnorm_gfx1250.csv).
Ratios below one mean regressions and are retained. N=128 ATOM small-row
cases have little benefit and some regressions (M=1: 2.046 -> 2.187us).
This is not an across-the-board speedup. N=6144 generic dispatch is unchanged;
its differences are run variation.

Baseline/optimized median cross-pass drift: 0.60% / 0.56%.
Maximum drift: 12.32% / 13.89%.
Drift is abs(t1-t2)/mean(t1,t2); per-shape values are in the CSV.
Sub-0.1us differences should not be interpreted as reliable speedups.

## Validation

* 136 parametrized cases: FP16/BF16, ordinary/Gemma weights, residual on/off,
  contiguous/padded rows, padding sentinels, widths 128/256/6144/8192,
  M=255/256/257 boundary and dual-input grid-y fallback.
* Profiler confirms 32x4 for N=128, original 512x16 for N=6144/M=128,
  and 256x24/interleave for N=6144/M=256.
* Both changed kernel translation units cross-compiled for gfx950.
  No gfx950 device execution is claimed.
* 32 FP8 grouped-quant fallback cases compared against the parent revision:
  widths 128/6144, residual on/off, ordinary/Gemma norm, group sizes 32/128,
  generic/fused paths. Quantized bytes, scales and residual outputs matched
  **exactly**. Each case/path ran in its own process; the fused path used its
  raw pybind entry with the canonical aiter tensor conversion and stream.
  A combined-path Python test aborted on the unmodified baseline with
  `q_epsilon is required`; this is not claimed fixed by the optimization.
  FP4 and per-token quant were not tested.
* During the timed ABBA runs, periodic GPU0 process monitoring found no
  external client. Sampling cannot rule out sub-second interference.

## Reproduce

Use an idle gfx1250 GPU and identical toolchain, with separate BASE and OPT
checkouts/JIT caches. Install normal aiter dependencies first.

    HIP_VISIBLE_DEVICES=0 GPU_ARCHS=gfx1250 PYTHONPATH="$BASE" AITER_JIT_DIR="$PWD/jit_base" python "$OPT/op_tests/bench_rmsnorm_gfx1250.py" --out before1
    HIP_VISIBLE_DEVICES=0 GPU_ARCHS=gfx1250 PYTHONPATH="$OPT" AITER_JIT_DIR="$PWD/jit_opt" python "$OPT/op_tests/bench_rmsnorm_gfx1250.py" --out after1

Repeat optimized then baseline with the same seed, and pool 18 samples.

    HIP_VISIBLE_DEVICES=0 GPU_ARCHS=gfx1250 PYTHONPATH="$OPT" python -m pytest -q "$OPT/op_tests/test_rmsnorm_gfx1250_dispatch.py"
    HIP_VISIBLE_DEVICES=0 GPU_ARCHS=gfx1250 PYTHONPATH="$OPT" python "$OPT/op_tests/bench_rmsnorm_gfx1250.py" --profile-only --out profile

The script saves every raw timing sample and metadata. It uses upstream
C++ JIT and does not bypass work, change precision, or detect benchmarking.
