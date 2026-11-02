#!/usr/bin/env python3
import csv,json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent
rows=[]
for mode in ('baseline','baseline_recheck','hybrid'):
 for c in (1,8,16,32,64,128):
  p=r/f'results/{mode}_c{c}_profile0.json'
  if not p.exists():continue
  d=json.loads(p.read_text())
  rows.append(dict(mode=mode,concurrency=c,completed=d['completed'],mean_tpot_ms=d['mean_tpot_ms'],median_tpot_ms=d['median_tpot_ms'],mean_ttft_ms=d['mean_ttft_ms'],median_ttft_ms=d['median_ttft_ms'],output_tok_s=d['output_throughput'],total_tok_s=d['total_token_throughput']))
if rows:
 with (r/'metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for x in rows:print(x)
profile=[]
for mode in ('baseline','baseline_recheck','hybrid_quant','hybrid'):
 for c in (16,64):
  for rank in range(4):
   p=r/f'analysis/{mode}_c{c}_rank{rank}'
   if not (p/'summary.json').exists():continue
   d=json.loads((p/'summary.json').read_text());kr=list(csv.DictReader((p/'kernels.csv').open()))
   q=[x for x in kr if 'FUSE_QUANT_1' in x['kernel']]
   profile.append(dict(mode=mode,concurrency=c,rank=rank,steps=d['decode_steps'],kernels_per_step=d['kernel_count']/d['decode_steps'],wall_ms=statistics.mean(x['wall_us'] for x in d['steps'])/1000,fused_quant_calls=sum(int(x['count']) for x in q),fused_quant_mean_us=sum(float(x['sum_us']) for x in q)/sum(int(x['count']) for x in q) if q else None))
(r/'profile_summary.json').write_text(json.dumps(profile,indent=2))
