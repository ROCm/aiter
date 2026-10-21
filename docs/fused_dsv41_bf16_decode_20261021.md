# DSV4.1 BF16-attention decode optimization — 2026-10-21

The FP8-cache decode pipeline retains BF16 Q, BF16 QK/PV, FP32 accumulation, and the existing RoPE/output quantization semantics. It now uses gfx950 packed scaled FP8-to-BF16 conversions for prefetched KV tiles. H128 FP8 calls without an extend CSR use two 64-head CTAs per query instead of one 128-head CTA; BF16-cache H128 calls retain their existing dispatch.

The baseline is the previously validated `combined` candidate, carried in commit `461560a7ae06dda3bf2f846330c59219dc907740`. The selected experiment is `native_split128`. Public interfaces and cache layouts are unchanged.

## Original six decode fixtures

MI355X GPU 6. Time includes Q RoPE, KV dequantization, attention, inverse RoPE and MXFP8 output cast. Layout preparation, metadata, cache writer and GEMM are excluded. PR #6675 is pinned at `b92f3cc9eeff2b5a5c56741a15da003dc5640d2e`; its full path also includes Q FP8 quantization and uses FP8 QK. Native row-packed KV is supplied directly to PR. Clocks were not locked.

| T | H | S | Old paged us | New paged us | Reduction | Old row us | New row us | PR full us |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 560 | 64 | 8192 | 142.84 | 131.37 | 8.03% | 148.19 | 138.16 | 113.85 |
| 560 | 64 | 32768 | 146.53 | 135.15 | 7.77% | 151.27 | 141.09 | 115.91 |
| 960 | 64 | 8192 | 179.56 | 161.38 | 10.12% | 184.05 | 168.28 | 178.51 |
| 960 | 64 | 32768 | 182.14 | 164.47 | 9.70% | 184.87 | 169.70 | 178.47 |
| 320 | 128 | 8192 | 235.61 | 208.80 | 11.38% | 245.26 | 219.94 | 154.46 |
| 320 | 128 | 32768 | 245.92 | 216.53 | 11.95% | 253.65 | 231.17 | 157.39 |

Timing uses six balanced orders, five event samples per visit, four calls per graph and eight replays per sample; the result is the median of visit medians. The final run contains 1,080 event samples and 216 poison/replay checks against saved references. New and old fused output/scale bytes match exactly on all six fixtures, with max/LSE checked at 1e-6. Three independent dense queries per case validate fused BF16 attention and three validate PR FP8-Q attention.

## Supplemental validation

- 219 native tests passed, including BF16 inputs, paged/legacy caches, ragged/invalid rows, RoPE, outputs and graph replay.
- 20 real pinned-vLLM writer checks have zero error; old low-level defaults also pass.
- All 65,536 FP8 byte/E8M0 scale-code combinations were compared between scalar and packed conversions on gfx950. Every non-NaN result, including signed zeros, matches bitwise. 256 NaN bit patterns differ while remaining NaN; NaN payload identity is not asserted.
- Five supplemental H128 shared-cache shapes (S=8192, K=640) match output/scale bytes exactly and pass another 60 saved-reference replay checks.

| T | Old us | New us |
|---:|---:|---:|
| 1 | 62.11 | 52.35 |
| 32 | 63.11 | 52.89 |
| 128 | 70.83 | 60.09 |
| 512 | 142.77 | 142.02 |
| 4096 | 1147.53 | 1102.69 |

## Profiling and tradeoffs

For the T560/H64 paged fixture, dynamic VALU instructions fall from 24,832,640 to 19,042,240; MFMA count stays at 2,867,200. The H64 code object uses 24 B/thread scratch instead of 32 B/thread, with 256 VGPRs and unchanged 67,584-byte LDS per CTA.

For T320/H128, the two-CTA path keeps the MFMA count at 5,898,240 but raises VMEM read instructions from 634,880 to 1,016,320 because each half reads KV. The measured latency improvement therefore trades additional KV traffic for finer CTA scheduling. It does not establish an occupancy increase or a benefit for every possible batch/cache shape.

The FP32-to-BF16 conversion candidate, H128 barrier removal, QK/PV commit timing and 64-head dispatch were evaluated separately. QK-tail/PV timing changes were not retained in the selected H128 path. The optimized BF16 path beats the PR full path on the two T960/H64 fixtures; PR remains faster on T560/H64 and T320/H128. PR and fused attention use different QK precision.

## Evidence and reproduction

Remote evidence: `/home/jun_chen2_qle/yzhou_aiter/decode-bf16-opt-20261021`.

Use `final_verified.json` for the final comparison. `final_compare.json` is marked invalid because its extension namespace reused the old fused module. The corrected harness gives every binary a unique namespace, checks module identity and retains all tensors backing low-level descriptors. A separate initial benchmark launch was retried after renaming a task script that shadowed Python's stdlib `profile`; neither issue was a kernel correctness failure.

The selected binary is `jit/native_split128/module_fused_dsv41_mxfp8.so`, SHA256 `3695e0d2b43d29fbed329d4d2a33f9ede626f6a762a8a6126cc08f5dc1176ca2`. The tested source and binary were checked again after final validation.

With the recorded container/repository dependencies and saved JIT artifacts, run `final_compare.py --candidate native_split128 --output <fresh-name>.json` under `HIP_VISIBLE_DEVICES=6`. `candidate.py`, the pytest XML, primitive probe, writer checks, raw event samples and profiler counters are retained alongside the report. Full ATOM/vLLM model-forward integration is outside this kernel benchmark.
