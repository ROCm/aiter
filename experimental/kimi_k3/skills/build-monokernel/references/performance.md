# Performance experiments and correctness evidence

## What launch fusion can and cannot save

Device launch latency is the time to execute an almost-empty kernel, distinct
from CPU API overhead. Graph capture reduces host submission overhead but does
not make each device launch free. Measure the empty-kernel cost on the target;
2–3 microseconds was the working motivation for K3, not a universal constant.

A useful accounting model is:

    gain = removed device launch gaps + removed intermediate traffic
           - added waits - lost parallelism - extra work/resource pressure

This is a hypothesis to test, not a latency estimator that can be applied without
a trace. Twenty removed launches cannot compensate for hundreds of microseconds
of slower expert math. A fused kernel may be slower while using fewer launches.

When batch rises, separately count routed rows, unique experts, padded expert
blocks, weight bytes, CTA iterations, reduction bytes, and register/LDS demand.
At small batch the removed launches can dominate; at larger batch memory-level
parallelism and expert throughput can dominate. Concurrency in a serving client
is not necessarily the physical batch of every decode iteration.

## Progressive fusion and matched ablations

Keep an independent unfused mathematical reference and the current serving
baseline. Use identical inputs, weights, routing, precision, workspace layout,
launch mode and sequence lengths for a fusion-only comparison.

Start at complete MoE, then compare front/expert/tail boundaries. Add the
post-attention norm/AttnRes and then output projection/TP reduction. For another
model choose boundaries from its graph, not K3's stage numbers. Measure at each
increment and retain the winning split. Test any precision change separately;
otherwise a faster quantized GEMM can be mistaken for a fusion benefit.

For each ablation save source revision, shape/TP, compile constants, register and
LDS use, spills, occupancy, launch count, per-rank samples, maximum-rank latency,
and correctness. Coarse stage timestamps help find waits, but one CTA's times do
not represent the whole grid. Full-grid instrumentation can perturb timing;
collect traces separately from clean performance runs.

If a concurrency is substantially behind the reference, stop the sweep. Tune one
representative batch (K3 used B2/B4 and later B8), then rerun protected batches.
Different batch-specific parameters or kernels are legitimate. Freeze retained
variants so JIT globals/import edits cannot silently alter their specializations.
Compare final selected variants, not abandoned intermediate tunings.

## gfx950 tricks worth testing, not blindly copying

| Technique | Why it can help | What can invalidate it |
|---|---|---|
| Explicit coherent publication/polling | Avoid repeated compiler-inserted global cache flush/invalidate sequences | Incorrect release/visibility proof |
| Vectorized coherent payload reads | Reduce instruction overhead while preserving visibility | Misalignment, masks, wrong ISA lowering |
| Buffer loads for immutable weights | Better addressing/instruction scheduling for regular tiles | Address range/layout constraints; not a semaphore-load replacement |
| Packed weight layout | Match MFMA operands without per-launch transposes/dequant | Wrong expert scale shuffle or changed library layout |
| LDS input staging | Reuse input tiles and improve operand delivery | Bank conflicts, barriers and occupancy loss |
| Moderate loop unrolling | More independent weight requests and instruction overlap | VGPR growth, spills, instruction-cache pressure |
| Smaller row tiles / more K splits | More producers in flight, less per-CTA work | Extra reductions and mailbox/CTA overhead |
| 256 to 512 resident CTAs | More parallel work if two CTAs/CU fit | Consumer starvation if residency falls to one |
| Batched tail reuse | Reuse shared/down/up work across multiple tokens | Extra padding/registers for small shapes |
| Wave-level publication | Reduce unnecessary CTA-wide synchronization | Missing participating waves or premature publication |
| Packed zero guards | Skip provably empty packed expert work cheaply | Misinterpreted encoding or skipped valid work |
| Partial norm statistics | Avoid rereading a full tensor for every output tile | Wrong reduction order or precision change |
| Remove unused quantization | Eliminate work whose outputs are not consumed | Another specialization still needs those outputs |
| Spare-CTA metadata and zeroing | Hide small independent work under the dense front | CTA role overlap, metadata consumer ABI changes |

K3's dense front moved from 64 output rows/two K splits to 32 rows/four splits;
that uses 188 producers per four-token group at TP8. The selected B2/B4/B8
front used unroll 14, tail rows 32 and 512 CTAs. Those values came from matched
experiments. Increasing unroll further was not monotonic. Sleep in polling loops
and smaller tail tiles did not win in earlier experiments; later changes to the
schedule changed which tile sizes worked. Do not infer independent gains from
combining individually favorable changes.

The explicit-coherence K3 ablation removed repeated `buffer_inv sc1` and
`buffer_wbl2 sc1` sequences and showed a large improvement. That is ISA and timing
evidence, not hardware-counter proof of an exact cache-level bottleneck. Preserve
this distinction when explaining root causes.

## Benchmark protocol when GPU execution is authorized

Use an allocation free of other GPU workloads. Do not stop another user's work.
Warm compilation and library dispatch before rank alignment and timing. Capture
repeated invocations in a graph to measure device work, choose enough iterations
to amortize events, and report multiple repeats and the slowest rank. Avoid
including first JIT, model loading, or host barriers in the device interval.
Alternate candidate/control order or use paired runs to detect drift. Save clean
raw measurements; a concurrent job contaminates results even if correctness passes.

Record empty-kernel and communication-only latency separately. Compare actual
collective sizes/topology. A single-GPU or single-rank microbenchmark cannot
establish TP performance, and TP4 performance cannot establish TP8 performance.

## Numerical and serving validation

Check routing IDs exactly, including ties, and check routing weights. Check
intermediate tensors to locate errors: projection, norm, latent, expert local
output, TP sums, final residual. Use structured and random inputs, real weights,
all relevant layers, edge routing patterns, and short/long residual history.

FP32 math is useful as a diagnostic reference, but the deployed baseline has
rounding boundaries too. Compare against both when needed. Tiny dense/router
rounding changes can alter top-k or amplify through low-bit expert quantization.
Do not loosen tolerances solely to pass a candidate; explain the error source and
report nRMSE/max error/ID disagreements. Operator checks do not replace lm-eval.

For metadata, test all tokens choosing the same distinct expert set, disjoint
sets, mixed overlap, extreme IDs, shuffled order, padding, and changed routes on
replay. Compare both sorted structures and reconstructed weighted expert output.
Repeated-ID cases outside the supported contract should be rejected or explicitly
handled, not counted as passing supported cases.

Exercise graph replay with changed inputs and allocator churn. Validate every
integrated layer/rank, mailbox errors, and generation canaries before performance
promotion. Run relevant model-quality evaluation when resources permit; report
it as unrun otherwise. Follow applicable repository/model evaluation preferences.

For vLLM comparisons preserve the recipe, ISL/OSL, seed, prompt count, warmups,
KV dtype, prefix cache/speculation settings, TP, graph settings and quantization.
Report vLLM's TTFT, TPOT, output token throughput, and total token throughput with
units. Total throughput counts prompt plus generated tokens; it is not generated
tokens/s. TTFT includes prefill/queue effects and may vary independently of TPOT.
Do not derive serving TPOT from a region microbenchmark or combine mismatched runs.
A few-percent target needs repeated uncontended evidence, not one favorable sample.
