#!/usr/bin/env python3
import csv,json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(r/'summarize_results.py')],check=True,stdout=subprocess.DEVNULL)
rows=[]
for c in (1,8,16,32,64,128):
 ds={mode:json.loads((r/f'results/{mode}_c{c}_profile0.json').read_text()) for mode in ('baseline','baseline_recheck','hybrid')}
 for mode,d in ds.items():
  assert d['completed']==4*c,(mode,c,d['completed'])
  assert d['total_output_tokens']==4*c*1024
 b,h=ds['baseline_recheck'],ds['hybrid']
 rows.append(dict(concurrency=c,baseline_tpot_ms=b['mean_tpot_ms'],hybrid_tpot_ms=h['mean_tpot_ms'],tpot_reduction_pct=100*(1-h['mean_tpot_ms']/b['mean_tpot_ms']),baseline_ttft_ms=b['mean_ttft_ms'],hybrid_ttft_ms=h['mean_ttft_ms'],baseline_total_tok_s=b['total_token_throughput'],hybrid_total_tok_s=h['total_token_throughput'],throughput_gain_pct=100*(h['total_token_throughput']/b['total_token_throughput']-1)))
with (r/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['\nSERVING RESULTS\n','Comparison uses the matched baseline_recheck control versus final hybrid.\n','Mean TPOT/TTFT in milliseconds; total throughput includes input+output.\n',' C | baseline TPOT -> hybrid | baseline TTFT -> hybrid | total tok/s baseline -> hybrid | TPOT reduction\n']
for d in rows:
 lines.append(f"{d['concurrency']:3d} | {d['baseline_tpot_ms']:.3f} -> {d['hybrid_tpot_ms']:.3f} | {d['baseline_ttft_ms']:.2f} -> {d['hybrid_ttft_ms']:.2f} | {d['baseline_total_tok_s']:.2f} -> {d['hybrid_total_tok_s']:.2f} | {d['tpot_reduction_pct']:.2f}%\n")
lines+=['\nAll runs, including medians and the original baseline:\n','mode,C,mean_TPOT,median_TPOT,mean_TTFT,median_TTFT,total_tok_s\n']
for d in csv.DictReader((r/'metrics.csv').open()):
 lines.append(','.join(d[k] if k in ('mode','concurrency') else f'{float(d[k]):.3f}' for k in ('mode','concurrency','mean_tpot_ms','median_tpot_ms','mean_ttft_ms','median_ttft_ms','total_tok_s'))+'\n')
lines.append('''
Each hybrid point is one completed run; controls were measured twice. These
are observations on this image/workload, not confidence intervals or broad
model-quality claims. Original c8 TTFT included a large tail; original c16 TPOT
also varied materially. All raw runs are retained, rather than selecting the
best result per concurrency. Profile latency is reported separately above.
''')
p=r/'REPORT.txt';s=p.read_text().split('\nSERVING RESULTS\n')[0];p.write_text(s+''.join(lines))
print(''.join(lines[:10]))
