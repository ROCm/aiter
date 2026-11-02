import csv
import json
from pathlib import Path

root=Path(__file__).parent
rows=[]
for mode in ('baseline','hybrid'):
 path=root/'results'/f'{mode}_client2_c16.json'
 if not path.exists():continue
 d=json.loads(path.read_text())
 assert d['completed']==64
 assert d['total_input_tokens']==65536
 assert d['total_output_tokens']==65536
 ids=[json.loads((root/f'{mode}-server-{stage}.json').read_text()) for stage in ('before-client1','after-client1','after-client2')]
 assert len({(x['pid'],x['start_ticks']) for x in ids})==1
 row={'configuration':mode,'reported_client':2,'concurrency':16,'completed':d['completed'],'server_pid':ids[0]['pid'],'server_start_ticks':ids[0]['start_ticks'],'same_server_for_both_clients':True,'source':str(path.relative_to(root))}
 for k in ('mean_tpot_ms','median_tpot_ms','mean_ttft_ms','median_ttft_ms','total_token_throughput','output_throughput','duration','total_input_tokens','total_output_tokens'):
  row[k]=d.get(k)
 rows.append(row)
out={'protocol':'One server launch per configuration; full client1 warmup, full client2 reported; no restart between clients.','rows':rows}
if len(rows)==2:
 b,h=rows
 out['tpot_reduction_pct']=100*(b['mean_tpot_ms']-h['mean_tpot_ms'])/b['mean_tpot_ms']
 out['throughput_gain_pct']=100*(h['total_token_throughput']/b['total_token_throughput']-1)
(root/'comparison-client2.json').write_text(json.dumps(out,indent=2)+'\n')
if rows:
 with (root/'comparison-client2.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
s=['Pro TP4 C16 rerun — second full client only','',out['protocol'],
   '1024 input/1024 output,64requests,seed0,ignoreEOS,request rate inf,real routing,BF16KV,FP8index,CUDA graphs,no profiler or debug comparisons. Same current upstream ATOM plus corrected unsplit integration for both arms; only ATOM_DSV4_MONO changes0->1.','',
   'Configuration | mean TPOT ms | median TPOT ms | mean TTFT ms | total tokens/s | same-server PID/start ticks']
for x in rows:
 s.append(f"{x['configuration']} | {x['mean_tpot_ms']:.6f} | {x['median_tpot_ms']:.6f} | {x['mean_ttft_ms']:.6f} | {x['total_token_throughput']:.6f} | {x['server_pid']}/{x['server_start_ticks']}")
if len(rows)==2:s+=['',f"Hybrid TPOT reduction:{out['tpot_reduction_pct']:.3f}%; total throughput change:{out['throughput_gain_pct']:.3f}%."]
s+=['','Client1 JSON/logs are preserved as warmup, not selected for the comparison. PID and start-time snapshots prove no restart across the two full clients. Prior22.274755ms baseline used the older checkout; this is a fresh same-source pair. No statistical-confidence claim from one reported client per arm. Scheduling timer probe inconclusive on gfx1250; see PROBE_NOTE.txt.']
(root/'REPORT.txt').write_text('\n'.join(s)+'\n')
print('\n'.join(s))
