from pathlib import Path
import json,re,datetime
root=Path(__file__).resolve().parent;results=root.parent/'concurrency-rollout'
rows=[];pending=[]
for m in [2,4,8]:
 row={'concurrency':m}
 for label,tag in [('native','batch-full-native'),('full','batch-full-fixed')]:
  p=results/f'{tag}-c{m}-round2.json'
  if not p.exists():pending.append(str(p));continue
  data=json.loads(p.read_text())
  assert data['completed']==m*8,(p,data['completed'])
  row[label]={'ttft_ms':data['median_ttft_ms'],'tpot_ms':data['median_tpot_ms'],'total_tokens_per_second':data['total_token_throughput'],'output_tokens_per_second':data['output_throughput'],'completed':data['completed'],'source':str(p)}
 if 'native' in row and 'full' in row:row['throughput_change_pct']=(row['full']['total_tokens_per_second']/row['native']['total_tokens_per_second']-1)*100
 rows.append(row)
accuracy={}
for label in ['baseline-c2','baseline-c4','baseline','fixed-c2','fixed-c4','fixed-c8']:
 files=list((root/f'gsm8k-{label}').rglob('results*.json'))
 if files:
  d=json.loads(files[-1].read_text());accuracy[label]={'correct':round(d['results']['gsm8k']['exact_match,strict-match']*32),'n':32,'source':str(files[-1])}
report={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pending' if pending else 'complete','model':'DSV4 Flash','parallelism':'single GPU TP1, no EP','scope':'one fused decode kernel per full layer including attention and routed/shared MoE; all43 layers','benchmark':'1024 input /1024 output, server once per candidate configuration, two clients per concurrency, use round2 only','ttft_tpot_statistic':'median','total_tok_s':'input plus generated output tokens per second','rows':rows,'accuracy':accuracy,'accuracy_limit':'32-example GSM8K subset, 3-shot, max512 generated tokens; not full validation. C8 strict score30/32 includes numerically correct16.00 vs16 formatting failure. C4 native/full both30/32 but different failed examples.','padding_fix':'Ignore graph padding via batch_id_per_q_token for compressor plans, identity indices, KV writes; mask regression tests and unmasked negative control saved.','prefetch':'Bounded nonspeculative TDM prefetch for QA/QB/cache; early TDM weight loads retained.','pending':pending}
(root/'RESULTS.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'status':report['status'],'rows':rows},indent=2))
