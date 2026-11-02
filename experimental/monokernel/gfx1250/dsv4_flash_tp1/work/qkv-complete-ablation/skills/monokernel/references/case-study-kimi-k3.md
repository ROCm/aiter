# Case study: Kimi-K3 decode on 8x MI355X (TP8), Oct 2026

Use this as a worked example of the method. All constants are specific to this model, stack and host.

## Model and baseline

- 93 layers: one dense layer, then MoE layers (896 experts, top-16 sigmoid routing with correction bias, a
  "latent" factored MoE: down-proj 7168 -> 3584, all-reduce, norm, expert MLPs in the latent, up-proj back), a shared
  expert (gate_up, SiTU activation, down), two attention kinds (KDA linear attention on most layers, MLA every few
  layers), and "AttnRes" residual mixing over stored block summaries instead of a plain residual add.
- Serving: vLLM, eager Python captured by full CUDA graphs (not torch.compile), speculative decoding (DSpark drafter)
  with k=7 at c=1 (8-row verify steps), k=5 at c=2-4, k=4 at c>=10 (240 rows at c=48).
- Baseline per-op MoE: AITER fused MoE (MXFP4 weights, sort + quant + stage-1 + stage-2 kernels); all-reduces:
  QuickReduce INT4 (lossy) and a custom 2-stage all-reduce; dense GEMMs: Triton / hipBLASLt.
- Reference monokernel: FlyDSL implementation in vLLM PR #60683 (K1 = attention-front launch, K2 = o_proj..MoE
  launch, 8-row steps only).

## Rung 0/1 at 1-16 rows: Gluon monokernel

- Started with a custom task table + `noinline` phases: S=1 168.8 us/layer vs FlyDSL 110. Switched to the reference's
  stage walk (every CTA walks all stages, bid-indexed work, phases inlined): par, then wins.
- Wins at S=1 (span): LDS-staged GEMV + one-round in-proj 113 -> 109.6; wave-local asm polls 109.1; tagged mixer
  outputs 106.4; stage walk 105.3; async-LDS prefetch raising in-flight bytes 256 -> 320 KB per CTA 97.6; removing a
  hold-back delay 93.3; one idle CTA publishing the top-16 91.1 (parity with FlyDSL 91); vectorized 8-peer push
  (5.7 -> 1.8-3.1 us exchange) 87.9-88.9, ahead of FlyDSL.
- K2 rewrite on the PR's ABI (8 rows): 178 -> 121.5 us vs FlyDSL 120.2 on random routing — but 6.7% slower end to
  end (4.61 vs 4.32 ms TPOT). Cause: real 8-token speculative steps route to ~24 distinct experts, not ~112; at that
  distribution FlyDSL 90.5 vs Gluon 104.6. Tuning at |U|=24 (chunk-split partials, skipping dead epilogue slots,
  publish-before-prefetch, per-CTA route tables) reached 97.0. Expert-up breakdown: MMA 0.6 us of 15.2; the rest was
  decode, route tables, weight issue latency, epilogue. A 2-vs-1 task imbalance cost ~3 us.
- Splitting K2 into Gluon pre -> AITER MoE -> Gluon post: 92.6 us vs fused 83.8 vs FlyDSL 75.9 (multi-process).
  Rejected.
- 16 rows (c=2): Gluon 16-row K2 132.8 us vs FlyDSL as two 8-row launches 149.8 in the harness, but 164.6 us in the
  server, and per-op serving (6.71 ms TPOT) beat both mono variants (6.92, 7.56). Only 15% of agentic c=2 steps were
  16-row, so the 8-row kernel's speed dominated.
- Bugs worth remembering: MFMA-result read in inline asm without `s_nop` (nondeterministic columns); load hoisting
  above a spin loop (zeros on first launch); a region sized for 8 rows overflowing at 16 with valid tags (hang that
  bisects pointed away from); spills at 8 rows faulting a persistent kernel.

## Rung 2 at 240 rows: library GEMM + library MoE + fused glue

- Per-op layer at 240 rows ~567 us: MoE ~215, collectives ~95-130, dense GEMMs ~110-150 (at only 1-2.5 TB/s), AttnRes
  2x20, top-k 15.
- Dense GEMM study: floors 6.5-11.6 us per GEMM; hipBLASLt ~13.5 us o_proj; hand Gluon ~16 (LLVM scheduler exposes LDS
  latency; per-CU in-flight cap); Triton sweep no better. Kept the library.
- Glue kernels (Gluon): `ar_attnres` (o_proj all-reduce + residual update + AttnRes + norm), `route_situ` (top-16 +
  shared-expert activation), `lat_ar_norm` (latent all-reduce + norm), `final_ar` merged with the next layer's
  attention-side AttnRes + input norm (pending partial across the layer boundary).
- Exact bf16 pull all-reduce over uncached IPC memory: 484 -> 33 us (plain stores for data, preloaded peer bases,
  pull all-gather, GEMM writing directly into the peer buffer, 224 owner units with 128 B segments).
- Harness: fused 458.9 vs per-op 531.9 us/layer. Server profile: 571 vs 565 us/layer — no gain. Reasons:
  (1) the harness per-op all-reduce cost 56 us vs 22 in the server; (2) the glue bypassed vLLM's MoE layer runner and
  fused top-k with the shared-expert activation, serializing the shared expert that the baseline overlaps on a side
  stream (~27 us net); (3) the MoE kernels ran ~12 us slower — identical code objects, output memory type irrelevant
  (controlled probe), so a workload difference: MoE time ~0.19 us per distinct expert, and the glue run's tokens
  differed (exact vs INT4 all-reduce numerics, lower speculative acceptance — flagged for live correctness checks).
- Per-kernel results (all-window, all-rank medians): `ar_attnres` 36.5 vs 45.9 and 39.3 vs 46.5 us replaced;
  `route_situ` 12.4 vs 18.2; `lat_ar_norm` ~1 us worse than the two kernels it replaced.
- Side-stream fix (split top-k and activation, shared expert on the aux stream): correct; harness gain only 4 us when
  started next to the bandwidth-bound router/latent GEMMs; starting after the latent GEMM like the baseline is the
  next experiment.

## Lessons this case contributed

1. Benchmark at the server's routing distribution; random inputs hid a 6.7% gap.
2. The full monokernel wins only while every op is small (<= 8 rows here); at 16 rows a stretched 8-row design lost
   to per-op; at 240 rows only glue fusion makes sense.
3. Harness gains must be confirmed against the server's own per-op costs.
4. Re-implement the stream schedule of any framework layer you bypass; never fuse across streams.
5. Peer-memory all-reduce performance is about store mode, wait-counter coupling, pull vs push and segment size.
6. Adopt the reference design first; spend effort on the differences that measurements point to.
