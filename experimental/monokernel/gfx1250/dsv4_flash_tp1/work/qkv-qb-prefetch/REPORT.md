# QB L2 prefetch and first-tile TDM, Flash C1

Implemented all requested staging in the **single-launch Q/KV preparation region**, not a full attention/MoE layer or an ATOM serving run. Starting point: qkv-load-poll fixed-order FP32 reduction, device-scope pure vector-load polling, QB N128/K128/unroll4, BF16 WMMA. Numerical operations and output boundaries are unchanged.

## Placement and mapping

- 256 CTAs, 4 wave32 waves each. QB FP8 weights are [32768,1024], 32 MiB, plus 2 KiB E8M0 scales (scales use existing loads).
- Thread `tid` in CTA `pid` prefetches byte addresses `QBW + pid*131072 + tid*1024 + line*256`, line=0..3. Each thread owns one 1 KiB QB row. Assuming the 256-byte cache line, the mapping covers all 131072 lines once. CPU enumeration is recorded in coverage.json; API checks base alignment.
- Four vector prefetch instructions **per wave**, 4096 dynamic wave instructions across the grid, issue all line hints. This is 32 MiB requested, not four instructions worth of completed HBM transfer.
- The 128 QA and 64 KV producer CTAs stage after their partial-result publication and group-counter arrival, before testing the last-arriver ticket. Thus the final CTA stages before RMSNorm; others stage before polling. The 64 idle producer CTAs stage directly before polling. No extra global rendezvous is inserted.
- TDM asynchronously loads each CTA's first [128,128] FP8 QB tile (16 KiB payload, 18 KiB padded LDS). It waits only after the existing grid poll, immediately before QB; block zero reads LDS, remaining seven blocks use normal loads. No redundant global read of block zero in TDM mode.
- Generated ISA confirms four `global_prefetch_b8` instructions on each mutually exclusive producer/idle path, `tensor_load_to_lds` before polling, and `s_wait_tensorcnt 0` only at QB consumption. See isa-verification.json and saved assembly. The compiler also emits an unrelated scope-SE kernarg prefetch.

## Hint encoding

Tested speculative RT (TH=0) and non-speculative RT (TH=1), both scope DEV. LLVM assembler uses the normal-load spelling `TH_LOAD_NT` for TH=1, but **for device-scope GLOBAL_PREFETCH_B8 it means non-speculative regular-temporal prefetch**, not non-temporal prefetch. Confirmed encoded TH bits with llvm-mc on gfx1250. AMD's [CDNA5 ISA section 4.1.1](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna5-instruction-set-architecture.pdf), printed page36, defines this distinction. Prefetch has no completion result and is not used as synchronization.

## Measured results

| Variant | Reused weights (us) | 16 rotating copies (us) |
|---|---:|---:|
| Single-launch baseline | 11.1833 | 12.8456 |
| L2 speculative | 11.2664 | 13.4159 |
| First-tile TDM | 11.0617 | 13.2007 |
| L2 speculative + TDM | 11.1709 | 13.1746 |
| L2 non-speculative | 11.2069 | 13.4711 |
| L2 non-speculative + TDM | 11.0522 | 13.3877 |

Only **round two** is reported. Round one is retained but discarded for the comparison. Each graph has128 serial invocations, five graph warmups,40 alternating-order event samples per round. One copy:38 MiB of QA/KV+QB weights. Sixteen copies:608 MiB; identical values at distinct addresses, a larger-working-set proxy rather than a claim of entirely cold HBM or actual multilayer serving. All variants are interleaved in the same benchmark process. Original four-way experiment is preserved in initial-benchmark; final six-way is scratch-benchmark.

Combined non-speculative L2+TDM reduces reused-weight time by about1.2%, but increases rotating-weight time by about4.2%. Speculative L2+TDM is essentially tied warm and about2.6% slower rotating. TDM alone is about1.1% faster warm and2.8% slower rotating. Small warm gains varied between the initial and final experiments; no robust general improvement is established. The options remain available, **disabled by default**.

Why the expected large win did not appear is not yet proven by hardware counters. Four issue instructions still request32 MiB of data. The short reduction/norm overlap window may be insufficient; the prefetch burst can compete with partial-result reads and QB demand loads. TDM removes only the first one-eighth of QB weight demand, and adds descriptor/LDS handling. These are hypotheses, not measured stall attribution.

Resources: baseline/L2-only289VGPR,8KiB LDS; TDM variants305VGPR,26KiB LDS. All zero spills/private bytes. No occupancy-loss claim is made solely from these counts.

## Correctness

All six variants passed268 varied inputs ×3 replays (804 checks each), plus10240 checked changing-input graph invocations each, including the earlier quantization-threshold counterexample. Q/KV relative-RMS gate0.001 unchanged; health/counters clean. See scratch-validation/summary.json and scratch-stress/results.json for exact maxima. Standalone API checks all six options and the two-launch control. No serving or tokens/s claim.

## Reproduce

Container vigokhal_dsv4_tp4, gfx1250, /work/qkv-qb-prefetch. Run ./run.sh while GPU is idle and with no concurrent persistent grids. Fixture contains necessary weights/input/rotary tables; no full checkpoint required. Sources, environment, raw timings, validation, assembly and LLVM IR are included. See README.md for API.
