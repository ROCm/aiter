from pathlib import Path
import json
r=Path(__file__).resolve().parent
c1=json.loads((r.parent/'modelwide/corrected-c1-summary.json').read_text())
rows=[]
for c in (1,2,4,8):
 for name in ('native','candidate'):
  if c==1:d=c1['native' if name=='native' else 'corrected_full']
  else:
   p=r/f"{'baseline' if name=='native' else 'qkv-hybrid'}-c{c}-round2.json"
   d=json.loads(p.read_text()) if p.exists() else None
  if d is not None:
   rows.append(dict(concurrency=c,path=name,scope='native' if name=='native' else ('full layer' if c==1 else 'QKV region only'),**{k:d[k] for k in ('completed','median_ttft_ms','median_tpot_ms','total_token_throughput','output_throughput')}))
(r/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n')
s='''# Flash model-wide experiments: saved 2026-10-21

Stopped at the user's request; no further experiments are running. Current image target: `rocm/fw-bringup:dsv4_tp4_mk_expts`. This is an experimental snapshot, not a production-qualified release.

Single GPU, TP1, 1024 input / 1024 output, maximum context2048. Report only client run2, after client run1 on the same server lifetime. TTFT/TPOT are medians. Total throughput includes input and output. Native means the installed ATOM native path with experimental fusion disabled, not a pristine historical image.

| Concurrency | Path | Completed requests | TTFT ms | TPOT ms | Total tok/s | Output tok/s |
|---|---|---:|---:|---:|---:|---:|
'''
for d in rows:
 s+=f"| {d['concurrency']} | {d['scope']} | {d['completed']} | {d['median_ttft_ms']:.2f} | {d['median_tpot_ms']:.2f} | {d['total_token_throughput']:.2f} | {d['output_throughput']:.2f} |\n"
s+='''
## What completed and what did not

C1: one fused kernel per layer across all43layers, including mHC, QKV, attention/reduction/output projection, and routed/shared MoE. Prefill, embeddings, final output head remain native. Corrected late weight binding, packed indexer cache stride, precise FP32 mHC statistic, matched attention rounding. Graph/eager smoke matched5/5; limited GSM8K32sample check native31/32 and fused31/32. This is not full accuracy qualification. C1 raw timings: `../modelwide/corrected-c1-client-round2.json` and `baseline-client-round2.json`.

C2 candidate: QKV-region-only fusion across43layers. Native attention, mHC and MoE. Both clients completed16requests each; the result is preliminary because the same implementation failed longer serving atC4/C8. C8 limited GSM8K32sample check passed30/32, matching native score, but it did not catch the later fault.

C4/C8: native baselines complete; NO valid candidate second-run performance. Graph candidate hit illegal memory access under 1K/1K serving. Do not report crashed/incomplete runs as speedups. Native-tail isolation also failed a longer test. All-native QKV custom-op control completed16requests atC8, suggesting the fused execution path or its graph interaction remains at fault; this does not prove the exact cause. Eager long test was stopped during its warmup at user request and has no result.

Serialized full-batch prototype is REJECTED: VGPR1024 with spills and barrier stalls. Runtime refuses `ATOM_DSV4_BATCH_SERIAL=1`. A genuinely batched full-layer implementation remains unfinished.

## Current batched implementation

`qa_batch.py`/`qkv_batch.py`: all M<=8 rows reuse each WMMA weight tile;256CTAs,192QA/KV projectors, splitK16; QA reduction matches AITER local8-FMA accumulation/ascending reduction. Early TDM of QB tiles overlaps QA. Four per-thread L2 prefetch instructions cover the32MiB QB weight matrix. QB uses scaled WMMA, followed by head/sequence-parallel Q norm/RoPE. KV scatter is included.

Fixed an independently real scatter bug: destination is `Dest[token_row]`, NOT `Dest[batch_id]`; batch_id only gates padding. That fix alone did not eliminate long-run faults. `ATOM_DSV4_BATCH_QKV_NATIVE_TAIL=1` runs native final Q/KV norm/RoPE/cache write after fused projections. `ATOM_DSV4_BATCH_QKV_NATIVE_CONTROL=1` uses native math inside the same opaque custom-op boundary. Both are diagnostic switches, not default benchmarks.

Microbenchmarks against actual ATOM kernels: `qa-batch-native-results.json`, `qkv-batch-native-results.json`. QA, quant, scales and QB matched on saved varied-input cases; final Q relativeRMS <=about1.53e-5. These local checks do not establish long-run serving correctness. See README for AITER sources/tricks studied.

## Resume tomorrow

1. Read `/home/vigokhal/MONOKERNEL_RESUME.txt` and this report. Restore image or extract complete archive, mount checkpoint at `/data` and project at `/work`.
2. Reproduce C4/C8 graph illegal access with `qkv-hybrid` below; retain native control and native-tail ablations. Check actual runtime metadata/cache strides and masked padded rows, lifetime/aliasing of custom-op outputs, graph capture/replay buffers, and barrier health counters. Capture inputs and compare native/candidate stage-by-stage on private state. These are hypotheses to investigate, not established root causes.
3. Finish eager long test (stopped during warmup), preferably begin with shorter output to localize first fault. `eager-server.sh` is level0/enforce-eager. Diagnostic serialized runs must never be compared as performance.
4. Only after correctness/stability, run C4/C8 client twice on one server; report second. Extend batched attention/mHC/MoE fusion after the QKV integration is stable. Avoid nonresident grid barriers; batch rows should share WMMA tiles rather than serially repeating C1.

## Launch commands

Image contains `/work` research artifacts and `/app/ATOM`, `/app/aiter`, `/app/scripts`. Checkpoint is external. Start container with ROCm devices, host IPC/network and sufficient shared memory as in existing environment. For example inside container:

```bash
# Native batch baseline, max sequences8 and captures[1,2,4,8]
HIP_VISIBLE_DEVICES=0 HTTP_PORT=8000 PORT=8001 SERVER_LOG=/work/concurrency-rollout/native-new.log bash /app/scripts/run_dsv4_flash_experiment.sh baseline
# C1 full-layer configuration (same supported context specialization)
HIP_VISIBLE_DEVICES=0 HTTP_PORT=8000 PORT=8001 SERVER_LOG=/work/concurrency-rollout/full-c1-new.log bash /app/scripts/run_dsv4_flash_experiment.sh full-c1
# Experimental QKV hybrid; C4/C8 long-run fault is unresolved
HIP_VISIBLE_DEVICES=0 HTTP_PORT=8000 PORT=8001 SERVER_LOG=/work/concurrency-rollout/hybrid-new.log bash /app/scripts/run_dsv4_flash_experiment.sh qkv-hybrid
# Separate shell, exactly two clients, report round2; change tag to avoid overwrites
HTTP_PORT=8000 bash /app/scripts/run_dsv4_flash_experiment.sh client resumed-hybrid 2
```

Set ONLY `HIP_VISIBLE_DEVICES=N`; setting both HIP and ROCR toN>0 caused double-remapping/no active device during one startup. Do not start two memory-heavy servers on the same GPU; one diagnostic startup failed from temporary overlap/OOM. Stale failed servers and clients were stopped.

Historical TP4 scripts remain `/app/scripts/serve_dsv4_mk_tp4.sh` and `/app/scripts/bench_dsv4_mk_tp4.sh`. No new TP4 validation was performed during this Flash session.
'''
(r/'REPORT.md').write_text(s)
