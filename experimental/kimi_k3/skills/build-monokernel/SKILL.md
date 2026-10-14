---
name: build-monokernel
description: Design, implement, debug, and tune resident GPU monokernels or hybrid fused pipelines, including cross-CTA mailboxes, peer-memory collectives, and library MoE stages. Use for model-layer fusion and launch-latency optimization, especially AMD gfx950 Gluon/FlyDSL ports.
---

# Build a GPU monokernel

Produce a correct, measurable fusion of a model's dependency graph. A monokernel
is one device launch containing multiple dependent operations; a hybrid keeps
large library operations between fused front/tail launches. Neither implies an
entire transformer layer. State the actual fusion boundaries.

Respect the task's execution constraints. Source-only work does not authorize
GPU queries, imports that initialize HIP, JIT compilation that loads a device,
server startup, or benchmarks. Do CPU checks and list deferred device evidence.
The procedures below describe future experiments, not permission to run them.

## Choose the architecture from evidence

1. Map the model's operations, tensors, dtypes, layouts, reductions, routing,
   residuals, TP/EP ownership, and stream dependencies. Read the actual reference
   implementation and serving dispatch; do not port a list of operator names.
2. Record device/compiler/library versions and the exact baseline recipe. Fix
   physical batch, TP, sequence lengths, quantization, and graph-capture mode.
3. When GPU work is authorized, prove coherent mailboxes and IPC collectives
   independently before composing the full graph. Port one dependency edge and
   check the generated instructions, then extend from MoE toward attention.
4. Compare a fully fused resident schedule with a hybrid. Keep a large GEMM/MoE
   outside the resident grid when it needs more independent memory requests,
   different occupancy, or a better quantized library implementation. Fuse
   routing, metadata, quantization, reduction, and residual work only where the
   consumer ABI and memory-ordering proof permit it.
5. Validate each increment and time matched ablations after every few stages.
   At the first substantial regression, stop the sweep and tune one representative
   batch. Retain separate specializations if one choice regresses another batch.
6. Integrate only supported shapes, prepare persistent state outside capture,
   and repeat operator, graph, all-layer, serving, and model-quality checks as
   appropriate. Record what passed separately from what remains unrun.

## Read the relevant references

- Before implementing an edge or porting ISA, read
  [synchronization.md](references/synchronization.md). It covers tagged payloads,
  separate payload/flag publication, IPC, epoch lifetime, and timeout diagnosis.
- Before writing a resident schedule or selecting a hybrid split, read
  [architecture.md](references/architecture.md). It includes routing metadata
  construction, consumer ABI requirements, occupancy, and graph ownership.
- Before experiments or interpreting a regression, read
  [performance.md](references/performance.md). It covers launch latency,
  progressive ablation, gfx950 tuning, numerical checks, and serving metrics.
- For concrete implementation anchors and measured lessons, read
  [kimi-k3-case-study.md](references/kimi-k3-case-study.md). Treat its dimensions,
  CTA IDs, instruction encodings, and tuning constants as case-specific.

## Required reasoning artifacts

Keep these alongside the code, at a depth proportional to the change:

- A dependency/ownership table: writer, reader, buffer, lifetime, readiness
  condition, memory scope, and reuse rule for every cross-CTA or cross-rank edge.
- A CTA assignment with the maximum resource footprint and resident-capacity
  calculation. Account for all specialized branches of the compiled kernel.
- A consumer ABI description for retained library kernels: padded rows, packed
  IDs, scales, intermediate layout, accumulation/zeroing, and output ownership.
- A reproduction recipe and evidence ledger distinguishing source checks,
  device correctness, profiling, uncontended performance, and model quality.

Do not claim four device launches from four Python calls. A library call may
launch sorting, quantization, zeroing, conversion, reduction, or copies. Read its
selected dispatch, constrain the supported ABI, and confirm with a trace once
GPU execution is allowed. CPU mocks can verify wiring, not device launch count.

Do not turn a local speedup into a universal tuning rule. Preserve a known-good
variant and its evidence; keep unsupported cases on the established path rather
than silently adapting assumptions about layouts, topology, or precision.
