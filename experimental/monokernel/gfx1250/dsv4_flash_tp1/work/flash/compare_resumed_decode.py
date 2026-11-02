from pathlib import Path
from collections import defaultdict
import csv,json,statistics
r=Path(__file__).parent;rows=[];summary={}
def group(k):
 if k=='fused':return 'fused_routed_moe'
 if '_moe_gemm_a8w4' in k or 'reduce_grouped' in k:return 'library_routed_moe_gemms_reduce'
 if 'dynamic_per_group_scaled_quant' in k:return 'group_fp8_quant'
 if 'mhc' in k.lower():return 'mhc'
 if '_pa_decode_sparse' in k:return 'sparse_attention'
 if 'gemm_mxfp8_preshuffle' in k or 'gemm_a8w8_mx128' in k or 'flydsl_mxfp8_' in k:return 'fp8_projections'
 return 'other'
for mode in ('resumed_baseline','resumed_final'):
 for c in (1,16,128):
  path=r/f'analysis/{mode}_c{c}'
  if not (path/'summary.json').exists():continue
  d=json.loads((path/'summary.json').read_text());n=d['decode_steps'];agg=defaultdict(lambda:[0,0.])
  for z in csv.DictReader((path/'kernels.csv').open()):
   a=agg[group(z['kernel'])];a[0]+=int(z['count']);a[1]+=float(z['sum_us'])
  assert sum(v[0] for v in agg.values())==d['kernel_count']
  summary[f'{mode}_c{c}']=dict(decode_steps=n,kernels_per_step=d['kernel_count']/n,mean_wall_us=statistics.mean(z['wall_us'] for z in d['steps']),mean_busy_union_us=statistics.mean(z['busy_union_us'] for z in d['steps']),mean_sum_us=statistics.mean(z['sum_us'] for z in d['steps']))
  for k,(cnt,dur) in agg.items():rows.append(dict(arm=mode,concurrency=c,kernel_family=k,calls_per_step=cnt/n,summed_us_per_step=dur/n,mean_kernel_us=dur/cnt))
if rows:
 with (r/'resumed_decode_kernel_families.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(r/'resumed_decode_comparison.json').write_text(json.dumps(summary,indent=2))
for x in rows:
 if x['concurrency']==1:print(x)
