# C8 QKV dependency ablations — 2026-10-21

Selected: local producer quantization + bounded pre-B2 QB L2 prefetch, split16 / output tile128. Retain B3: QA-only readiness did not provide a measured win. No serving/full-layer integration performed.

Times are microseconds for the entire eight-row group: input quant, QA/KV projection, split reduction/norm/quant, QB, Q norm/RoPE. Current KV produced; production cache scatter is outside this fixture. No new group6/SWA timing.

Reported value: median of four GPU-specific second-round medians, all variants interleaved on each GPU; first round discarded. This is microbenchmark timing, not a client result. See per-GPU data and samples in final0..3.

|Variant|K splits|Output tile / projection CTAs|Time us|GPU range us|
|---|---:|---:|---:|---:|
|ATOM native|—|—|27.539|27.496–27.623|
|Previous fused group|16|128 / 192|20.524|20.377–20.685|
|Local quant only|16|128 / 192|18.532|18.408–18.674|
|Local quant + early prefetch (selected)|16|128 / 192|18.242|18.160–18.643|
|Above + QA-only readiness|16|128 / 192|18.538|18.381–19.657|
|All three changes|8|128 / 96|18.995|18.925–19.239|
|All three changes|8|64 / 192|18.782|18.601–20.449|
|All three changes|4|128 / 48|21.039|20.981–21.165|
|All three changes|4|32 / 192|21.313|21.097–21.458|
|All three + FP32 atomics|16|128 / 192|26.079|24.586–27.421|
|All three + FP32 atomics|8|64 / 192|22.236|21.875–22.658|
|All three + FP32 atomics|4|32 / 192|23.084|22.860–23.167|

## Validation

All variants were compiled and checked for zero spills before launch; selected uses {'regs': 149, 'spills': 0, 'lds': 121344}. Compiled ISA/LLVM and metadata are retained. Non-speculative prefetch uses tensor descriptor bounds. Assembly confirms prefetch instructions before B2 atomic/poll, scaled WMMA, and device-scope buffer polling.

Stress tests use32 seeds: zero input, random inputs, saved live activation plus perturbations, and row-varying scales. Each case runs two32-launch graph replays without resetting counters/epochs (2048 candidate invocations per variant). Check output poison overwrites, Q/KV/raw outputs, FP8 activations/scales, and health flags. Exact comparisons are to the previous fused path, not a claim of model-wide exactness. Worst relative RMS to native is reported below.
- stress-early: 64 checks, Q exact vs previous=True, FP8/scales exact=True, Q native relative RMS max=2.02233477e-05, KV previous RMS max=0.
- stress-s4: 64 checks, Q exact vs previous=True, FP8/scales exact=True, Q native relative RMS max=2.02233477e-05, KV previous RMS max=0.
- stress-s8: 64 checks, Q exact vs previous=True, FP8/scales exact=True, Q native relative RMS max=2.02233477e-05, KV previous RMS max=0.
- stress-atomic: 64 checks, Q exact vs previous=True, FP8/scales exact=True, Q native relative RMS max=2.02233477e-05, KV previous RMS max=0.
- stress-sem: 64 checks, Q exact vs previous=True, FP8/scales exact=True, Q native relative RMS max=2.02233477e-05, KV previous RMS max=0.

## Findings and rejected trials

- Local quant removes B1 and global FP8 input handoff. Each producer uses TDM BF16 input and locally converts/scales complete128-element groups before WMMA.
- Earlier prefetch provides a smaller improvement. It is issued after projection stores, before the B2 synchronization. Existing first-QB-tile early TDM is retained.
- QA-only readiness uses an8-arrival device counter plus256 per-CTA monotonic epoch slots; release producers/coherent load polling. It is correct on tested graphs but adds epoch bookkeeping, and is not selected.
- Split8/BN64 and split4/BN32 maintain192 projection CTAs. Split8 comes close; split4 loses despite preserving CTA count. Smaller split means more K work and quantization per CTA, while narrow N tiles provide less output reuse.
- Atomic initialization is included inside the fused launch before a barrier. No outside memset is hidden. Initial ordinary loads of the reused atomic scratch produced large errors on the initializing QA finisher CTAs. Device-coherent buffer loads fixed the tested errors; retain this fix. Initial invalid timings in gpu3 are diagnostic only; use atomic-fixed and final0..3 for valid atomic results.
- Atomics do not beat explicit reduction even at split8. They require initialization and contested updates; individual costs were not separately timed.
- No inference should be made that the regional speedup translates directly to full-model tokens/s. Full-layer residency and dual-forward validation remain required.

## Reproduce / integration

Use the existing vigokhal_dsv4_tp4 image environment, /home/vigokhal/dsv4-monokernel mounted as /work, model under /data. One worker per GPU; check ownership first. Run final_sweep.py with HIP_VISIBLE_DEVICES=0 and OUTDIR=/work/c8-qkv-deps/repeat0. Repeat on other GPUs in independent processes. Run stress.py --variant early (or sem/s8/s4/atomic). Existing common.py records the second round. Do not run build.py: it is the initial source-generation record, not a final build script.

Selected qkv_candidate.qkv_batch defaults LOCAL=True, PRE=True, SPLITS=16, BN=128, SEM=False, ATOMIC=False. Selected path needs no Ready allocation; SEM=True requires Ready[257] int32 initialized once, then reused on sequential graph invocations. Original argument signature otherwise retained, with optional experimental constexpr parameters appended. Scratch P remains[16,M,1536]. Preserve all production BatchId/Dest/cache masks and early prefetch bounds. Compile the full monokernel and verify zero spills and one resident CTA per CU before launching; standalone resource numbers do not establish full-layer residency.

The archive includes this experiment, Python dependencies, and the saved checkpoint-derived fixture. It still requires the matching ROCm/Triton/AITER/ATOM environment and model configuration.
