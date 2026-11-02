# Current configuration: keep L2 prefetch and all-weight TDM

User requested both mechanisms remain enabled. Default region.QKVRegion now uses all-weight TDM, early QB staging, and non-speculative device-scope L2 QB prefetch. Setting prefetch=False gives the TDM-only ablation; the direct-load version is a measurement control. This is a development choice, not selection solely by best measured time.

## Second-round measurements

| Path | Reused weights (us) | 16 rotating copies (us) |
|---|---:|---:|
| global_control | 11.002559 | 12.781312 |
| tdm_late | 10.598516 | 13.083016 |
| tdm_early | 10.359726 | 13.146078 |
| tdm_early_l2 | 10.789273 | 13.350758 |

Current requested configuration:10.789273us reused weights,13.350758us with16 rotating copies. In this same run, TDM-only early staging was10.359726us and13.146078us; the L2 hint currently adds cost. It stays enabled as requested.

## Historical comparison

Original full-layer-c1/regions/REPORT.md reports31.842us unfused and33.589us fused for QKV projections + norms + latent quant + RoPE (stages5..10). These were vector-math prototype controls, not native AITER/ATOM baseline. They excluded initial input quantization and used summed profiler GPU kernel durations, second round,30 graph replays with restored pre-region inputs. Original source/data remain unchanged.

The current QKV region uses BF16 WMMA, fixed-order FP32 split-K and fused quant/norm/rotary stages, and also includes initial input quantization. Current measurement is batched graph event timing,128 calls/graph,5 warmups,40 interleaved samples per round,2 rounds, second round only. Thus10.79us versus31.84/33.59us describes historical progress, not a controlled same-harness speedup. The old unfused prototype was not rerun today.16-copy608MiB working set is a proxy for different layer addresses, not actual multilayer serving. No tokens/s or end-to-end claim.

## Implementation and validation

QA/KV weight tiles and all8 QB weight tiles use TDM through LDS. QB double buffering starts before polling. Each CTA issues4 vector prefetch instructions per wave covering its128KiB portion of QB, at producer completion before last-arriver RMSNorm or before polling for other CTAs. Small vectors/scales, norm/reduction vectors and synchronization retain direct accesses.

All4 variants passed268 input cases x3 replays and10240 changing-input graph invocations each, with the unchanged0.001 relative-RMS criterion, clean counters and health. API checked global_control and both early/late TDM with default prefetch enabled. See JSON for exact metrics. ASM/LLVM and resource counts saved in scratch-benchmark. No serving integration, image update or git push. GPU idle after completion.

Reproduce ./run.sh in /work/qkv-all-tdm-l2 in the existing vigokhal_dsv4_tp4 container, on an idle gfx1250 with256 available CUs and no concurrent persistent grids. Source, fixture, assembly and raw results are self-contained. See README.md.
