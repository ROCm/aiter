# Porting checklist: monokernel / hybrid for a new model

Work through the sections in order. Sections 1-3 need no GPU; do them before asking for GPU time.

## 1. Model survey (fill this table from the framework's code, not the model card)

| Item | Record |
|---|---|
| Layers and layer types | count; which layers differ (dense vs MoE, attention kinds, layers writing extra state) |
| Per-op graph of each layer type | op, input/output shapes per token, dtype, weight format (bf16 / fp8 / MXFP4 + scale layout / shuffled layouts) |
| Parallelism | TP / EP / DP / context parallel sizes; which ops are sharded how; every collective (type, bytes per token, exact or quantized) |
| Residual structure | plain add, pre/post norms, multi-stream residuals or residual mixing; what state persists across layers |
| Routing (MoE) | experts, top-k, scoring (softmax / sigmoid + bias), grouping, renormalization, scaling, shared experts |
| Attention | kind (MHA/GQA/MLA/linear/SSM), KV-cache layout and dtype, per-token state for speculative rollback |
| Between layers | hidden states captured for drafters / aux heads; anything the framework reads as a tensor |
| Execution mode | torch.compile + piecewise graphs vs eager + full CUDA graphs; custom-op registration needs |
| Side streams | what the framework overlaps (shared experts, projections) and under what token threshold |
| Library kernels used | per op, the exact kernels a call launches at each bucket (read the dispatch and tuned tables) |

## 2. Workload and buckets

- Target serving recipe: concurrency points, speculative method and k per point, sequence lengths, quantization.
- Rows per step per point = concurrency x (k + 1); padded capture sizes the framework uses.
- Distribution of step widths in the real workload (replay or log); prefill vs decode share.
- Realistic routing statistics per bucket (distinct experts per step), measured from the server.

## 3. Baseline profile per bucket

- Server trace -> per-layer window table (kernel, stream, start, duration; wall / sum / union).
- Classify: big ops, glue, collectives; floors for each; library efficiency vs floor.
- Stream overlap present in the baseline and what it hides.
- Prize estimate per rung (launches and tails removed, collectives improved, overlap kept).

## 4. Design artifacts (write them down before code)

- Rung per bucket and the boundary list.
- Stage list with CTA ranges and overlaps; per-stage work units and their count vs grid size at the realistic
  distribution (equal tasks per CTA).
- Dependency / ownership table for every cross-CTA and cross-rank edge (writer, reader, region, size from max rows,
  readiness tag/flag, memory scope, reuse rule).
- Region layout derived from max rows with non-overlap asserts; epoch / tag scheme; timeout and health record.
- Resource budget per specialization (VGPRs, spills = 0, LDS, residency).
- For hybrids: consumer ABI of each library kernel; which producer writes directly into peer buffers; side-stream
  schedule to reproduce.

## 5. Build order

1. Unit-test the mailbox protocol and the cross-GPU exchange in isolation (correctness, then latency and bandwidth
   vs `amd-gpu-kernel-perf` numbers).
2. Port stages one at a time from one end (e.g. MoE back toward attention), each verified against a torch reference
   of its published intermediates; compile-only resource check and ISA review per stage.
3. Whole-layer check vs the framework's ops (rel error, cosine), all ranks identical, second forward OK.
4. Multi-process harness benchmark at each bucket with realistic routing; compare with the reference kernel if one
   exists, per stage via timelines.

## 6. Integration

- Build key -> compiled kernel; compile during warm-up with exact-dtype stand-ins.
- Eligibility guard (shapes, dtypes, weight formats, layer types, parallel config) with per-op fallback; log which
  layers and widths engaged; counters per path.
- Epoch bump captured in the graph; persistent buffers allocated outside capture.
- Residual / hidden-state materialization where the framework needs tensors.
- Side streams reproduced; nothing fused across streams.
- Patch or plugin shipping; launch script fails hard if the patch fails.

## 7. Acceptance

- Accuracy: speculative acceptance equal to baseline within noise; live side-by-side checks for a few steps; model
  quality eval if numerics changed (e.g. exact vs quantized all-reduce).
- Performance: median TPOT / ITL per bucket vs the baseline from the same session; profile confirms the expected
  per-layer savings; report harness, trace and benchmark numbers separately.
- Evidence ledger: what was verified on device vs only by static reasoning; what remains unrun.

## Questions to ask the customer early

- Which concurrency points and speculative settings matter (the bucket list), and the latency vs throughput goal.
- Weight formats they will serve (a guard that only accepts bf16 weights excludes fp8-quantized projections).
- Whether exact vs quantized collectives are acceptable (numerics change routing and acceptance).
- Framework version pin and whether a plugin, a patch, or an upstream PR is the delivery vehicle.
