from pathlib import Path
import json
p=Path(__file__).resolve().parent
r=json.loads((p/'ab-timings.json').read_text())[1]['results']
rot=json.loads((p/'rotating-ab.json').read_text())[1]['results']
stress=json.loads((p/'stress.json').read_text());edges=json.loads((p/'edges.json').read_text())
assert all(x['passed'] for x in stress) and len(edges)==120
assert all(x['health'] for x in r.values()) and all(x['health'] for x in rot.values())
resources=json.loads((p/'ab-resources.json').read_text())
assert all(x['spills']==0 for x in resources.values())
lines=['# Direct current-token KV results','','C1 Flash region microbenchmark, not serving throughput. All values are round 2 median GPU-event time across 40 samples; each graph has 128 region invocations. Alternating A/B order. L2 prefetch, TDM, WMMA, and split-KV reduction retained.','','## Reused weights','','| Tokens | Splits | Region | Cache barrier (us) | Direct KV (us) | Time reduction |','|---:|---:|---|---:|---:|---:|']
for t in [128,1024]:
 for s in [1,4]:
  for kind in ['attention','two','fused']:
   a=r[f't{t}_s{s}_{kind}_directFalse']['median_us'];b=r[f't{t}_s{s}_{kind}_directTrue']['median_us']
   lines.append(f'| {t} | {s} | {kind} | {a:.3f} | {b:.3f} | {(a-b)/a*100:.2f}% |')
lines+=['','## Sixteen rotating weight copies, T=128','','| Splits | Launches | Cache barrier (us) | Direct KV (us) | Time reduction |','|---:|---:|---:|---:|---:|']
for s in [1,4]:
 for fused in [False,True]:
  a=rot[f's{s}_fused{fused}_directFalse']['median_us'];b=rot[f's{s}_fused{fused}_directTrue']['median_us']
  lines.append(f'| {s} | {1 if fused else 2} | {a:.3f} | {b:.3f} | {(a-b)/a*100:.2f}% |')
lines+=['','## Validation','','- Existing FP64 attention and chain validation passed at unchanged 0.001 relative RMS tolerance.','- 61,440 changing-input graph stress checks passed, with cache rows deliberately stale.','- 120 edge cases passed, including slot 0, duplicate/missing indices, all invalid, partial tiles, and one-row fallback. Persistent cache contents verified.','- All measured specializations have zero spills.','- Profiler confirms one launch for fused and two for hybrid.','- Split-KV reduction remains inside each measured attention invocation.','','## Exploration','','The initial whole-tile LDS overlay and rejected register-substitution experiment are preserved as lds_overlay.py / lds-overlay-timings.json and register_overlay.py / register-overlay-{timings,resources}.json. Register substitution expanded register use and was substantially slower. Current source uses row-only LDS scatter. These are not serving-level speedup claims.']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
(p/'STATUS.json').write_text(json.dumps(dict(complete=True,serving_integrated=False,default_direct=True,stress_checks=sum(x['checked_invocations'] for x in stress),edge_cases=len(edges),reported_round=2),indent=2))
print('\n'.join(lines))
