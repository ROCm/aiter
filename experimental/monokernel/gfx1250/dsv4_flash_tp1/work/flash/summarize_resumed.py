from pathlib import Path
import csv,json
r=Path(__file__).parent;rows=[];comparison=[]
for c in (1,8,16,32,64,128):
 pair={}
 for arm in ('resumed_baseline','resumed_final','resumed_no_moe'):
  p=r/f'results/{arm}_run2_c{c}_profile0.json'
  if not p.exists():continue
  d=json.loads(p.read_text());assert d['completed']==4*c,(p,d['completed'])
  row=dict(concurrency=c,arm=arm,client_run=2,source=p.name,**{k:d[k] for k in ('mean_tpot_ms','median_tpot_ms','mean_ttft_ms','total_token_throughput','output_throughput','completed')})
  rows.append(row);pair[arm]=row
 if 'resumed_baseline' in pair and 'resumed_final' in pair:
  b=pair['resumed_baseline'];h=pair['resumed_final']
  comparison.append(dict(concurrency=c,baseline=b,candidate=h,tpot_reduction_pct=100*(1-h['mean_tpot_ms']/b['mean_tpot_ms']),throughput_increase_pct=100*(h['total_token_throughput']/b['total_token_throughput']-1)))
if rows:
 with (r/'resumed_serving_metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(r/'resumed_serving_comparison.json').write_text(json.dumps(comparison,indent=2))
for x in comparison:
 b=x['baseline'];h=x['candidate'];print(f"C{x['concurrency']}: TPOT {b['mean_tpot_ms']:.3f}->{h['mean_tpot_ms']:.3f}, TTFT {b['mean_ttft_ms']:.3f}->{h['mean_ttft_ms']:.3f}, total tok/s {b['total_token_throughput']:.1f}->{h['total_token_throughput']:.1f}, TPOT reduction {x['tpot_reduction_pct']:.2f}%")
