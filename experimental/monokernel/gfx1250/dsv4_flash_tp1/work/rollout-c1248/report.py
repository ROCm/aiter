from pathlib import Path
import json,statistics,hashlib,collections
r=Path('/work/rollout-c1248') if Path('/work/rollout-c1248').exists() else Path('/home/vigokhal/dsv4-monokernel/rollout-c1248')
micros={}
for c in [1,2,4,8]:
 for ratio in [0,4,128]:
  p=r/f'final-micro-{c}-{ratio}.json'
  if p.exists():micros[c,ratio]=json.loads(p.read_text())[0]
lines=['# Flash C1/C2/C4/C8 rollout','', 'Microbenchmark times are GPU microseconds on synthetic layer fixtures at position 1023, with the serving region bodies. Previous and updated monokernel paths are compared; these are not native ATOM unfused timings. Repeated paired graph timings, second round only.','', '| C | SWA previous | SWA updated | CSA previous | CSA updated | HCA previous | HCA updated |','|---:|---:|---:|---:|---:|---:|---:|']
for c in [1,2,4,8]:
 vals=[]
 for ratio in [0,4,128]:
  d=micros.get((c,ratio));vals.extend([f'{v["second_us"]:.3f}' for v in d['resources']] if d else ['pending','pending'])
 lines.append('| '+str(c)+' | '+' | '.join(vals)+' |')
lines+=['','Each isolated group below is previous → updated, in microseconds. Unchanged groups are retained; no prefix-time subtraction.','', '| Group | C1 | C2 | C4 | C8 |','|---|---:|---:|---:|---:|']
rows=[(0,0,'Attention mHC pre + input layout'),(4,1,'CSA compressor/index'),(128,1,'HCA compressor'),(0,2,'Input quant + QKVA + norm/RoPE + QB'),(0,3,'SWA attention + split-KV reduce'),(4,3,'CSA attention + split-KV reduce'),(128,3,'HCA attention + split-KV reduce'),(0,4,'Inverse RoPE + quant + OA/OB'),(0,5,'Attention mHC post + FFN mHC pre'),(0,6,'Routing + shared/routed MoE + sum + final mHC post/layout')]
for ratio,g,name in rows:
 vals=[]
 for c in [1,2,4,8]:
  d=micros.get((c,ratio));grp=next((x for x in d['groups'] if x['group']==g),None) if d else None
  t=grp['rounds'][1]['median_us'] if grp else None
  vals.append(f'{t["0"]:.3f} → {t["1"]:.3f}' if t else 'pending')
 lines.append('| '+name+' | '+' | '.join(vals)+' |')
lines+=['','End-to-end: Flash, one GPU per server, TP1, all 43 layers; 1024 input / 1024 output, C×8 requests. One server start per configuration; two client runs per concurrency; only round 2 shown. TTFT/TPOT are medians; total tokens/s includes input and output.','', '| C | Configuration | TTFT ms | TPOT ms | Total tok/s | Output tok/s |','|---:|---|---:|---:|---:|---:|']
e2e=[]
for c in [1,2,4,8]:
 for name in ['native','previous','selected','replica']:
  p=r/f'{name}-c{c}-round2.json'
  if not p.exists():continue
  d=json.loads(p.read_text());row=dict(c=c,configuration=name,ttft_ms=d['median_ttft_ms'],tpot_ms=d['median_tpot_ms'],total_toks=d['total_token_throughput'],output_toks=d['output_throughput'],completed=d['completed'])
  e2e.append(row);lines.append(f'| {c} | {name} | {row["ttft_ms"]:.3f} | {row["tpot_ms"]:.3f} | {row["total_toks"]:.2f} | {row["output_toks"]:.2f} |')
(r/'REPORT.md').write_text('\n'.join(lines)+'\n');(r/'E2E.json').write_text(json.dumps(e2e,indent=2))
# Prefill event timings grouped by client round and actual batch shape.
events=[]
for p in r.glob('prefill-*.jsonl'):
 for l in p.read_text().splitlines():
  try:events.append(json.loads(l))
  except json.JSONDecodeError:pass
buckets=collections.defaultdict(list)
for d in events:buckets[d['tag'],d['scheduled_bs'],d['tokens']].append(d)
prefill=[]
for (tag,bs,tokens),ev in sorted(buckets.items()):
 prefill.append(dict(tag=tag,batch=bs,tokens=tokens,n=len(ev),gpu_median_ms=statistics.median(e['gpu_ms'] for e in ev),cpu_enqueue_median_ms=statistics.median(e['cpu_enqueue_ms'] for e in ev),monokernel_calls=sum(e['monokernel_calls_during_prefill'] for e in ev)))
(r/'PREFILL_EVENTS.json').write_text(json.dumps(prefill,indent=2))
print(json.dumps(e2e,indent=2))
