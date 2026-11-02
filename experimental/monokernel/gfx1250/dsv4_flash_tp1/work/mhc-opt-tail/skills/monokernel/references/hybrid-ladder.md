# The hybrid ladder: from one launch per layer to fused glue between library kernels

A monokernel is not one design; it is the top rung of a ladder. As the number of rows a decode step processes grows
(concurrency × (speculative tokens + 1)), the big operators (dense GEMMs, MoE expert GEMMs, attention over long
context) become large enough that a tuned library backend beats anything you can schedule inside a resident grid,
while the small operators (norms, residual mixes, activations, routing, quantization, all-reduces) stay
latency-bound. The right design moves down the ladder as rows grow, and a production integration usually ships
several rungs, selected by row count.

## Contents
1. The rungs
2. The cost model that decides the rung
3. Choosing the rung: procedure
4. Fusing the glue (rungs 2-3)
5. Preserve what the baseline overlaps
6. Measured crossovers (Kimi-K3 on 8x MI355X, as an example)
7. Anti-patterns

## 1. The rungs

| Rung | Shape | When it wins | Typical rows |
|---|---|---|---|
| 0. Full monokernel | One persistent launch per layer (or per few layers); every op is a stage; cross-CTA and cross-GPU edges via tagged mailboxes in peer memory | Every op is small: weights stream once, each stage is latency-bound, launch gaps and collective round trips dominate | 1-16 |
| 1. Split monokernel | 2-3 persistent launches per layer at natural sync points (e.g. attention-front K1, MoE-back K2) | Same as 0, but one giant kernel spills or cannot be specialized per stage; boundaries placed where a global sync exists anyway | 1-16 |
| 2. Hybrid with library MoE / GEMM | Library kernels for the operators that have outgrown the grid; fused "glue" launches between them | GEMM/MoE M large enough for the library to reach bandwidth/MFMA roofline; glue still many tiny kernels plus collectives | ~32-512 |
| 3. Per-op baseline | Framework's default kernels | Prefill / large batches; glue is a small fraction of the step | 512+ |

Rungs are chosen per row-count bucket, not per model. Keep rung 0 for the buckets where it wins and add a hybrid for
larger buckets instead of stretching one kernel across all of them. Example: the FlyDSL monokernel shipped only 8-row
steps; at c=2 (16 rows) the per-op path ran and was faster than both monokernel variants.

## 2. The cost model

Estimate each op's floor and its fixed cost before choosing:

- Launch / boundary cost: ~2-2.5 us for an empty kernel on MI355X (2.38 us measured), plus a "tail" at every
  boundary: the next kernel waits for the slowest CTA of the previous one and, for collectives, the slowest rank.
  In CUDA-graph replay the host cost disappears but the GPU-side drain/fill does not.
- Memory floor: bytes / ~5.5-6 TB/s achievable HBM (not the 8 TB/s spec). A 22 MB weight ~6.5 us, 51 MB ~11.6 us,
  each including ~2 us of launch-and-ramp.
- Compute floor: FLOPs / ~1.5-1.7 PF sustained dense bf16 MFMA (not the 2.5 PF peak), fp4/fp8 scaled MFMA higher.
- Collective floor: an exact bf16 all-reduce of 3.4 MB over 8 GPUs by pull ~33 us; INT4-quantized ~35-39 us
  (lossy); each extra cross-GPU round trip ~1-5 us of latency on top.
- Resident grid limits: a persistent grid of one CTA per CU caps in-flight memory requests (~32 KB per CU of reads in
  flight on gfx950) and fixes occupancy for every stage; a library kernel can pick tile/occupancy per op. This is why
  a large GEMM inside the monokernel loses to hipBLASLt once M grows.

Rule of thumb: an op belongs in the library once its floor is several times the launch-and-tail cost and the
library reaches within ~1.2x of its floor at that M. An op belongs in a fused launch while its floor is comparable to
the launch cost or it is a collective whose latency you can overlap with independent work.

## 3. Choosing the rung: procedure

1. Measure the baseline per-op step at the target buckets (rows = concurrency × (k+1) for speculative decoding).
   Use a profiler trace from the real server, not a microbenchmark: build the per-layer window table (kernel, stream,
   duration) with the serving-perf skill's `trace_windows.py`.
2. Classify every kernel: big (GEMM/MoE/attention core), glue (elementwise, norm, routing, quant, small reductions),
   collective. Sum each class. Example at 240 rows (Kimi-K3): MoE ~215 us, collectives ~95-130 us, dense GEMMs
   ~110-150 us, AttnRes 2x20 us, top-k 15 us of a ~567 us layer.
3. For each big op compare the library time with its floor. If the library is within ~1.2x of floor, leave it.
   (At M=240 hipBLASLt was at ~13.5 us for o_proj against a ~6.5 us floor; a hand-written Gluon GEMM reached ~16 us
   and a Triton sweep found nothing better than vLLM's; the GEMM study was parked.)
4. The prize for fusing glue is (glue kernels' time + their boundaries) minus the fused kernel's time, minus any
   overlap you destroy. Compute it before writing code. Then check what the fused kernel will wait on: a fused
   all-reduce still pays the slowest rank.
5. Prototype in a multi-process harness that matches serving (one process per GPU, CUDA graph, realistic routing),
   then confirm in the server with a profile. Expect the harness to overstate the gain (see section 6).

## 4. Fusing the glue

Fuse at the boundaries of the big ops: everything between two library calls becomes one launch when the data
dependencies allow it.

- Typical fused glue units in a transformer decode layer:
  - Output-projection all-reduce + residual update + (residual mix / AttnRes) + post-attention norm (+ quantization
    for the next GEMM).
  - Router top-k + routing weights (+ quantization / sort metadata if the MoE ABI allows).
  - Partial-output all-reduce + norm between two halves of a factored MoE (latent down -> AR -> norm -> up).
  - Final all-reduce + the next layer's input-side residual mix + input norm (crosses the layer boundary; pass the
    un-reduced partial to the next layer as a pending value and materialize it only where the framework needs the
    tensor, e.g. the last layer or hidden states captured for a drafter).
- Let producers write straight into the collective's input buffer: point the GEMM's `out=` (or the MoE's `output=`)
  at the peer-visible buffer so the all-reduce owners pull from it directly. This removed the push stage
  (41 -> 35.5 us). Check the library honors `out=` without an extra copy, and which backend `out=` selects
  (`torch.matmul(..., out=)` chose hipBLASLt over vLLM's Triton split-K GEMM: 14.4 vs 16.8 us).
- Compute anything that does not depend on the collective before waiting on it (e.g. the block-source logits of a
  residual mix), so the round trip overlaps work.
- Keep numerics at least as good as the baseline: the exact bf16 pull all-reduce matched fp32 (rel 2.6e-3) where
  the INT4 quick-reduce baseline was lossy. Different numerics change the model trajectory, which changes routing and
  speculative acceptance; see the confounder section of the measurement reference.
- Library kernels launch more kernels than their Python call suggests (sorting, quantization, zeroing, copies).
  Read the dispatch to know the real boundary list and the consumer ABI (padding, packed ids, scale layout,
  accumulate-vs-overwrite output).

## 5. Preserve what the baseline overlaps

The baseline is often faster than its kernel sum because it runs independent work on a second stream. Find this
before designing: in the trace, compare per-window sum of kernel time, union of kernel intervals, and wall time.

- Kimi-K3 at 240 rows: vLLM's fused-MoE layer runs the shared expert (gate_up -> activation -> down, ~40 us) on
  `aux_stream()` when tokens <= a threshold; it hides behind top-k + MoE sorting/quant (~40 us). Baseline per layer:
  sum 592 us, union 554, wall 565 (11 us of cross-stream idle) = ~27 us net overlap.
- The first glue integration called the MoE library directly (bypassing the layer that owned the stream logic) and
  fused top-k (routed chain) with the activation (shared chain). Both serialized the two chains; the glue's ~20 us of
  kernel savings were cancelled. Rules:
  - Never fuse two ops that the baseline runs on different streams; a fused kernel that consumes both chains is a
    join point.
  - When you replace a framework layer, re-implement its stream schedule, not just its math.
  - Place side-stream work beside latency-bound kernels (routing, sorting, small elementwise), not beside
    bandwidth-bound GEMMs: starting the shared expert alongside the router/latent GEMMs recovered only 4 us of the 27.
  - Under CUDA graphs a side stream must fork from and rejoin the capture stream (`aux.wait_stream(main)` ...
    `main.wait_stream(aux)`); make sure every tensor used on the side stream outlives the rejoin.

## 6. Measured crossovers (Kimi-K3, TP8, MI355X; examples, not rules)

| Rows | Result |
|---|---|
| 1 | Full monokernel K1+K2: FlyDSL 60.3 us K2, Gluon 65.6; per-op loses clearly |
| 8 (c=1, k=7) | Monokernel wins: e2e 4.32 ms TPOT (FlyDSL) vs per-op baseline ~5.2 ms; split K2 (Gluon pre -> AITER MoE -> Gluon post) 92.6 us vs fused 83.8 vs FlyDSL 75.9: boundaries lost the prefetch overlap and added tails |
| 16 (c=2) | Per-op 6.71 ms TPOT beat Gluon 16-row monokernel 6.92 and FlyDSL-as-two-8-row-halves 7.56; most agentic c=2 steps are still 8 rows, so the 8-row kernel's speed dominated the c=2 benchmark |
| 240 (c=48, k=4) | Library GEMM (hipBLASLt) + library MoE (AITER) + fused glue: harness 458.9 vs 531.9 us/layer, but server parity at first (571 vs 565) until stream overlap is preserved; the harness's per-op path was more expensive than the server's (custom AR 56 vs 22 us) |

Takeaways: the full monokernel wins only while every op is small; at 16 rows a row-specialized kernel stretched
to twice its design point lost to per-op; at hundreds of rows the gain from fusing glue is real but modest (tens of
us per layer) and easily eaten by lost overlap.

## 7. Anti-patterns

- Splitting a fused kernel to "use the library MoE" at small rows (the library is no faster there and each boundary
  costs a tail). Use a split only as a diagnostic to locate a slow stage.
- Comparing a fused harness against a harness-built per-op path and calling the difference the gain; always confirm
  against the real server's trace at the same bucket.
- Tuning at one bucket and shipping for all: keep separate specializations per bucket and the per-op fallback.
- Attributing a library kernel's slowdown to your integration without checking the workload: MoE time scales with
  the number of distinct experts the batch routes to (~0.19 us/expert for one MoE stage at 240 rows); two runs with
  different token trajectories route differently.
