# Generated gfx1250 assembly analysis

Use the exact accepted and baseline dumps under `final-benchmark/`, not an approximate offline compile.

## Work mapping

Original A/G projection: BN32, split-K8, BK128, four waves. Grid384: Q-A256 CTAs, KV128. Each CTA loops over four K128 blocks. Four WMMA instructions appear in the loop, so each wave executes16 WMMA instructions. The N32 tile wraps the four-wave WMMA N mapping; extra waves perform redundant matrix work.

Accepted: BN128, split-K16, BK128, four waves. Grid192: Q-A128 CTAs (8 N tiles ×16 splits), KV64 (4×16). Each CTA processes K256. The compiler unrolls both K128 blocks and emits16 BF16 WMMA instructions per wave. Four waves cover distinct N portions. There is still M16 matrix padding for a single useful request row.

Estimated WMMA instruction executions from grid, wave count and ISA loops: original384×4×16=24,576; accepted192×4×16=12,288. This is static work accounting, not a hardware counter measurement.

## ISA comparison

These are **static instruction-site counts in the projection/fused dump**, not timing estimates. Original counts inside its loop execute four times. Normalization branches also mean not every static site executes on every CTA. Baseline A's separate reduction kernel is not counted in this table.

| Property | Original projection A | Old fused G | Optimized separate projection | Accepted fused split16 |
|---|---:|---:|---:|---:|
| K-loop headers | 1 | 1 | 0 | 0 |
| WMMA sites | 4 | 4 | 16 | 16 |
| Barrier signal sites | 6 | 10 | 0 | 6 |
| LDS load sites | 36 | 35 | 0 | 4 |
| LDS store sites | 10 | 12 | 0 | 11 |
| VGPRs | 128 | 130 | 221 | 214 |
| LDS bytes | 8192 | 8192 | 0 | 8192 |
| SGPR/VGPR spills | 0/0 | 0/0 | 0/0 | 0/0 |
| Private scratch bytes | 0 | 0 | 0 | 0 |

The important reduction is **operand** LDS traffic inside the repeated K loop. The accepted fused kernel still uses LDS for output-layout conversion, counter-ticket broadcast, and norm reduction.

Original conversion sequence: `v_cvt_scale_pk8_f32_fp8`, `v_pk_mul_f32`, `v_cvt_pk_bf16_f32`. Accepted conversion: `v_cvt_scale_pk8_bf16_fp8 ... scale_sel:8`, followed by `v_wmma_f32_16x16x32_bf16`. The FP8 checkpoint is retained; it is not expanded to BF16 storage.

Data publication uses `global_atomic_add_f32 ... scope:SCOPE_DEV`. The accepted output layout reduces the static FP32-atomic sites to two, versus eight for the otherwise comparable BN128 direct-accumulator layout. Instruction-site reduction is not fewer mathematical partial sums: split16 still contributes 16 partials per output channel.

Before the group completion RMW, the ISA contains explicit load/store-counter drains and a CTA barrier. Group completion uses `global_atomic_add_u32 ... th:TH_ATOMIC_RETURN scope:SCOPE_DEV`. Communication loads/stores remain device scope. There is no global polling loop, system-scope writeback, or `global_wb` in the accepted fused path.

The old G protocol needs 384 per-tile completion increments plus 48 per-group increments; the accepted mapping needs 192 group increments. It also removes the SS mailbox and the intermediate writeback of BF16-rounded tile values. The last Q/KV CTA reads the completed FP32 accumulator directly, rounds to BF16, computes norm, and resets its group.

## Diagnostic stage timestamps

`timeline.py` uses `s_sendmsg_rtn_b64 ... MSG_RTN_GET_REALTIME`, not unsupported CDNA `s_memrealtime`. HIP reported a 100 MHz wall-clock counter. The instrumented split16 median durations were about 1.04 µs projection/publication, 0.44 µs completion signaling, and 0.60 µs norm/reset. These include timer and instrumentation effects; the headline 2.733 µs value comes from the uninstrumented second-round CUDA-event graph benchmark.

## Why rejected alternatives did not help

- Direct loads with `.cg` remove LDS shuffles but expose an unfavorable load pattern. Cached operand reads and scale-conversion changes must accompany the layout change.
- Split-K1 exposes too few CTAs and makes each CTA run a long K loop: ~14us for the original N32 path versus ~4.6us at split8.
- BF16-expanded weights increase traffic/storage and were slower.
- Offline 16×32 and K128 weight tilings did not improve latency in the tested variants.
- TDM bulk LDS staging adds descriptor/wait/LDS work; explicit normal/speculative L2 prefetch issued too close to consumption did not pay for its overhead. This is a result for these implementations, not a claim that those hardware features cannot help other regions.
- A one-wave norm or one-wave output publisher saves some synchronization but loses parallelism or adds layout work.
- Parallel scratch reduction generated many extra cross-wave barriers (up to 98 static sites), and was substantially slower.
- Native FP8 WMMA is smaller/faster to issue but failed the expanded numerical gate; its timings are not accepted results.
