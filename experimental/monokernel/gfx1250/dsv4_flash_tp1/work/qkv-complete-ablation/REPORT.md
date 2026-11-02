# Complete Q/KV preparation, Flash C1

## Result

The requested ablation now includes input quantization, QA/KV projection and split-K reduction, first weighted RMSNorms, Qlatent quantization, QB projection, final per-head Q RMSNorm, and Q/KV RoPE. All projections use BF16 WMMA with FP32 accumulation and checkpoint FP8/E8M0 storage.

Second complete timing round, median microseconds per invocation:

| Working set | Separate stages (8 launches) | Two fused regions (2 launches) | Full region (1 launch) |
|---|---:|---:|---:|
| 1 weight copies (38 MiB) | 15.246 | 10.190 | 10.797 |
| 16 weight copies (608 MiB) | 16.717 | 12.721 | 13.328 |

The two-region path reduces latency33.16% for the repeated layer0 working set, versus the optimized separate-stage control. One launch reduces latency29.18%. For16copies the reductions are23.90% and20.27%. The16copies contain identical weights at different addresses, not16different layers, and do not assert a forced cold cache. Scale/gamma tables remain shared. Complete-region weight bytes are38MiB percopy.

This is an isolated single-GPU C1 microbenchmark on actual Flash layer0 projection weights. The baseline is the optimized custom WMMA separate-stage control, NOT native ATOM/AITER serving. No TPOT/TTFT/tok/s claim follows from these timings. No attention, KVcache publication, MoE, mHC, or output projection is included.

## Layout and synchronization

Recommended two-launch path:
1.192QA/KV CTAs:12outputtiles of128columns x16Ksplits. Each locally quantizes its two128-element input blocks, then WMMA. FP32 atomics combine splits. The final Qgroup CTA normalizes1024values and publishes FP8 Qlatent+8scales. The final KVgroup CTA normalizes512values and rotates final64channels. Both reset their counters/accumulator inside the invocation.
2.512QB CTAs:64columns each, no split-K, full K1024. Eight CTAs jointly produce one512-channel head. After BF16 rounding/store and explicit drain, the eighth ticket holder computes that head's RMSNorm and RoPE.64head counters reset before kernel completion. No spinning or resident-grid restriction in this path.

Full-region version:256resident CTAs, of which first192 perform QA/KV; all participate in one bounded device-scope grid barrier. QB uses128columns/CTA, one tile perCTA, four tickets perhead. No device-side launches. Requires256available CUs and independent state per concurrent stream. A nonzero health flag means invalid result. All per-invocation state resets are timed; persistent state allocation is excluded.

Device-scope coherent-L2 communication retains explicit completion drains and CTA synchronization. It does not rely on coherence alone for ordering. A stage barrier and different QB tiling account for part of the one-launch/two-launch difference; timings do not isolate a pure barrier cost.

## Standalone stages

Second-round separately measured latency: extended QA region3.929us; QB+headnorm+RoPE6.065us. Standalone input quant1.285us, already included via inline quant in extended QA, not an extra launch in fused variants. These isolated numbers must not be summed as a substitute for whole-chain timing. The previous2.733us measurement excluded input quant, latent quant, QB and RoPE; its scope is smaller.

## Validation

Independent FP64 projections on the exact quantized operands, BF16 projection rounding, FP32 norms and BF16 rotary output. Same relative-RMS gate <0.001 (0.1%) separately for Q and KV; threshold was not loosened.

268inputs x3replays x3variants =2412output pairs checked, with changing rotary tables. Allpassed. Across accepted two/single variants max Q relativeRMS0.00009161 (0.009161%); KV0.000124712 (0.012471%). Also10240changing-input graph invocations per fused variant, every output checked; no failure, counters/accumulators reset, healthzero. Standalone API separately checked for100graph replays. Results are in final-validation/, stress/, api-validation.json.

Inputs include a saved live activation, reconstructed prior bank inputs, zeros/constants and256reproducible random/perturbed scale cases. This does not establish full-model quality or GSM8K. Rotary tables are controlled fixtures, not claimed to be the serving YaRN cache.

## Compiled resources

| Kernel | VGPR | LDS bytes | Spills | Private bytes |
|---|---:|---:|---:|---:|
| Extended QA |112|8192|0|0|
| QB64+norm+RoPE |86|2048|0|0|
| Complete256CTA/QB128 |115|8192|0|0|

Actual AMDGCN and LLVM IR are saved in final-benchmark/. Earlier115VGPR single smoke variant used192CTAs; final uses256. Arithmetic uses WMMA instructions and packed scaled FP8-to-BF16 conversion. No native FP8 WMMA acceptance claim.

## Exploration and exclusions

Initial192CTA single-launch layout had uneven QB work (some CTAs executed two tiles). Balancing at256CTAs improved it. Keeping input quant as its own launch was slower than locally repeating quant inside QA producers. QB64 won in the two-launch path; QB128 won with256resident CTAs.

Exploratory benchmark/ and benchmark2/ BN256/512 results are NOT valid comparisons: the initial scalar weight-scale load did not handle tiles spanning multiple128-column scale groups. This was corrected in qb.py before finalvalidation; finaltimings use only validated BN64/128, which were unaffected. Do not use the exploratory12.94us single/BN256 observation. Initial FP8 broadcast compilation failures and logs are retained for reproducibility; final uses integer bitcast broadcast.

README.md documents reproduction and ABI. No serving integration, image publication, or branch push was performed.
