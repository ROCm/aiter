from pathlib import Path
import json,csv
r=Path('/work');rows=[]
for mode in ('round2_control','round2_candidate'):
 for c in (64,128):
  for warm in (False,True):
   name=mode+('_warm' if warm else '')
   p=r/f'results/{name}_c{c}_profile0.json'
   d=json.loads(p.read_text())
   rows.append(dict(mode=mode,concurrency=c,warmup=warm,**{k:d.get(k) for k in ['mean_tpot_ms','median_tpot_ms','p99_tpot_ms','mean_ttft_ms','median_ttft_ms','p99_ttft_ms','total_token_throughput','output_throughput','duration','completed']}))
with (r/'round2/serving_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
comp=[]
for c in (64,128):
 a=next(x for x in rows if x['mode']=='round2_control' and x['concurrency']==c and not x['warmup']);b=next(x for x in rows if x['mode']=='round2_candidate' and x['concurrency']==c and not x['warmup'])
 comp.append(dict(concurrency=c,control=a,candidate=b,tpot_reduction_pct=100*(1-b['mean_tpot_ms']/a['mean_tpot_ms']),throughput_increase_pct=100*(b['total_token_throughput']/a['total_token_throughput']-1)))
(r/'round2/serving_comparison.json').write_text(json.dumps(comp,indent=2))
a=json.loads((r/'round2/round2_control_batch_completions.json').read_text());b=json.loads((r/'round2/round2_candidate_batch_completions.json').read_text());validation={}
for c in ('64','128'):
 mismatches=[dict(index=x['index'],control=x['text'],candidate=y['text']) for x,y in zip(a[c],b[c]) if x['text']!=y['text']]
 validation[c]=dict(requests=len(a[c]),exact_text_matches=len(a[c])-len(mismatches),mismatches=mismatches)
(r/'round2/batch_validation.json').write_text(json.dumps(validation,indent=2))
print(json.dumps(comp,indent=2));print('Batch validation:',[(c,v['exact_text_matches'],v['requests']) for c,v in validation.items()])
