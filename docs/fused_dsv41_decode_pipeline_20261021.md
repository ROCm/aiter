# DSV4.1 BF16 decode: KV conversion and staging experiments, 2026-10-21

Selected candidate: `rotate_startup`, based on `0adf8daac99f17e418ce6679aa431f36ebc1709b` (`native_split128`).

The existing AMD producer path loads raw FP8 KV into registers, converts directly into BF16 LDS, and overlaps the next tile prefetch with current-tile QK. It already avoids a raw-FP8 LDS round trip. The selected change rotates each packed E8M0 byte into scale exponent bits 23:30 for gfx950 `cvt_scalef32_pk_bf16_fp8`. This instruction ignores the scale sign and mantissa, so byte extraction and scalar zero/NaN construction are unnecessary.

The first packed KV tile now uses the same prefetch/commit conversion as subsequent tiles, removing the scalar FP8-to-FP32, multiply, and BF16 conversion sequence from the prologue. The preceding round exhaustively checked this scalar-to-packed conversion over 65,536 FP8/E8M0 combinations: every non-NaN output was bit-identical; 256 NaN payloads differed while remaining NaN. NaN payload preservation is not asserted for this first-tile change.

Q and both QK/PV matrix multiplies remain BF16 with FP32 accumulation. Existing interfaces, cache layouts, head dispatch, RoPE, and output quantization are retained. Existing CTA synchronization and the order-sensitive PV scheduling constraints are retained. FP4 support and dedicated producer-wave specialization are not implemented by this change.

## Six original decode fixtures

MI355X GPU 6; unlocked clocks; microseconds. H64 uses K640 and H128 uses K1152. Every value below was measured in one final comparison with unique extension module namespaces and retained tensor storage. The baseline is the immediately preceding BF16 decode revision, not the older `combined` candidate.

| T | H | S | Old paged | New paged | Reduction | Old row-packed | New row-packed | PR full |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 560 | 64 | 8192 | 130.96 | 115.95 | 11.47% | 138.15 | 120.74 | 114.51 |
| 560 | 64 | 32768 | 135.02 | 120.51 | 10.75% | 141.39 | 123.70 | 115.45 |
| 960 | 64 | 8192 | 160.73 | 144.84 | 9.88% | 167.56 | 148.13 | 177.31 |
| 960 | 64 | 32768 | 163.72 | 146.84 | 10.31% | 168.91 | 150.17 | 178.53 |
| 320 | 128 | 8192 | 207.90 | 190.46 | 8.39% | 218.63 | 195.32 | 154.37 |
| 320 | 128 | 32768 | 216.20 | 198.93 | 7.99% | 226.47 | 206.89 | 157.25 |

Fused timing includes Q RoPE, selected KV dequantization, BF16 attention, inverse RoPE and MXFP8 output cast. PR #6675 at `b92f3cc9eeff2b5a5c56741a15da003dc5640d2e` includes Q RoPE, Q MXFP8 quantization, FP8-QK/BF16-PV attention, inverse RoPE and output cast. PR receives native row-packed KV; compare against our row-packed column for matched layout. Both paths exclude layout conversion, GEMM, metadata construction, cache writer and complete model forward.

## Rejected staging experiments

- `rotate_validity`: compressing validity into four wave ballots passed 219 tests, but its additional register and shift costs regressed against `scale_rotate`. Paged H64 scratch rose from 20 to 32 B/thread.
- `rotate_rawlds`: stash prefetched raw bytes/scales/validity in the unused BF16 slot, restore after QK, then synchronize before overwriting with BF16. It passed 219 tests and lowered paged scratch from 24 to 8 B/thread versus the parent, but added LDS traffic and one CTA barrier per prefetched tile. It also ends VMEM overlap early to complete the stash. This is a staging tradeoff experiment, not a port of FlashMLA TMA/warp specialization.

| Quick fixture | Baseline paged | Raw-LDS paged | Latency increase |
|---|---:|---:|---:|
| b140_q4_h64_s8192_k640 | 132.64 | 143.99 | 8.55% |
| b320_q3_h64_s8192_k640 | 165.68 | 177.81 | 7.32% |
| b80_q4_h128_s8192_k1152 | 208.51 | 242.43 | 16.27% |
| b80_q4_h128_s32768_k1152 | 217.31 | 254.41 | 17.07% |

FlashMLA inspiration should be interpreted at the pipeline level: producer/consumer overlap, compact scale handling, and slot lifetime. Our inspected FlashMLA snapshot has WG1 gather/dequant work and warp 10 coordinate/scale preparation; warp numbering varies by revision. NVIDIA TMA, warp groups, and AMD wave64/LDS/MFMA have different costs. This experiment does not establish that all dedicated producer-wave designs would be slower.

## Profiling

Dynamic paged-kernel counts below are medians of the final five selected dispatches under rocprofv3. Profiling timings are excluded from latency claims.

| Fixture index | Variant | VALU | LDS | VMEM reads | MFMA |
|---:|---|---:|---:|---:|---:|
| 0 | baseline | 19042240 | 4972800 | 530880 | 2867200 |
| 0 | rotate_startup | 16835840 | 4979520 | 512960 | 2867200 |
| 0 | rotate_rawlds | 17384640 | 5611200 | 530880 | 2867200 |
| 4 | baseline | 36180480 | 10147840 | 1016320 | 5898240 |
| 4 | rotate_startup | 31897600 | 10155520 | 995840 | 5898240 |
| 4 | rotate_rawlds | 32688640 | 11491840 | 1016320 | 5898240 |

Code-object resources and instruction summaries are recorded in `isa/summary.json`. LDS allocation remains 67,584 B/CTA and the active packed 64-head kernel remains at 256 VGPRs. A scratch reduction is not an occupancy claim.

## Correctness and reproducibility

- 219 native tests passed for the selected binary; the fixture suite covers BF16/FP8 inputs, legacy/paged cache addressing, invalid/ragged rows, RoPE and graph replay.
- The scale primitive probe covers all 256 FP8 bytes, 256 E8M0 codes, four code-byte positions and four unrelated-bit patterns: 1,048,576 combinations, zero bit mismatches including NaNs against the preceding scaled conversion.
- All six final fixtures have exactly matching fused output/scale bytes; max/LSE agree at 1e-6. Independent FP32 dense attention validates three queries per fixture for BF16 fused and three for PR FP8 Q.
- 20 real pinned-vLLM writer cases pass with zero error. Five H128 shared-cache sizes also match output/scale bytes exactly.
- Final timings use six balanced visit orders, five event samples per visit, four calls per graph and eight replays per sample. There are 1,080 event samples and 216 poison/replay checks against saved prevalidated outputs in the six-fixture comparison; shared-cache checks add 60 replays.

| Shared-cache T (H128, S8192, K640) | Old paged us | New paged us |
|---:|---:|---:|
| 1 | 52.39 | 45.99 |
| 32 | 52.95 | 46.59 |
| 128 | 59.98 | 52.58 |
| 512 | 142.29 | 126.22 |
| 4096 | 1102.87 | 984.36 |

Evidence directory: `/home/jun_chen2_qle/yzhou_aiter/decode-pipeline-opt-20261021`.

Selected binary: `/home/jun_chen2_qle/yzhou_aiter/decode-pipeline-opt-20261021/jit/rotate_startup/module_fused_dsv41_mxfp8.so`; SHA256 `af30da5d542569d15241f0c3c894ce33091af8d8b983ed9eb15d62552843df1b`.

Run `final_compare.py --candidate rotate_startup --output <fresh-name>.json` with the recorded container, dependencies, saved JIT artifacts and `HIP_VISIBLE_DEVICES=6`. Candidate generator, primitive probe, pytest XML, writer checks, paired candidate comparison, raw events and profiler counters are preserved. No matched B200 measurement was performed in this round.
