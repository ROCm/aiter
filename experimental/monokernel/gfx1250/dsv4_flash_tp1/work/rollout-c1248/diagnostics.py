import json,time,subprocess,sys,urllib.request
from pathlib import Path
root=Path('/work/rollout-c1248');name,port=sys.argv[1:]
while not all((root/(n+'.clients.done')).exists() for n in ['native','previous','selected','replica']):time.sleep(2)
for con in [1,2,4,8]:
 for rnd in [1,2]:
  tag=f'prefill-only-{name}-c{con}-round{rnd}'
  (root/f'prefill-{name}.tag').write_text(tag)
  cmd=[sys.executable,'-m','atom.benchmarks.benchmark_serving','--model=/data/DeepSeek-V4-Flash','--backend=vllm',f'--base-url=http://localhost:{port}','--dataset-name=random','--random-input-len=1024','--random-output-len=1','--random-range-ratio=1',f'--num-prompts={con*8}',f'--max-concurrency={con}','--request-rate=inf','--ignore-eos','--seed=0','--percentile-metrics=ttft,tpot,itl,e2el','--save-result','--save-detailed',f'--result-dir={root}',f'--result-filename={tag}.json']
  with open(root/(tag+'.log'),'w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
(root/(name+'.prefill-only.done')).touch()
while not all((root/(n+'.prefill-only.done')).exists() for n in ['native','previous','selected','replica']):time.sleep(2)
# Fixed token IDs, exact same prefill workload. No decode token generated after first output.
responses=[]
for con in [1,8]:
 for i in range(5):
  (root/f'prefill-{name}.tag').write_text(f'fixed-prefill-{name}-c{con}-{i}')
  if i==3:(root/f'prefill-{name}.trace').touch()
  tokens=[100+j%1000 for j in range(1024)]
  payload=dict(model='/data/DeepSeek-V4-Flash',prompt=tokens if con==1 else [tokens for _ in range(con)],max_tokens=1,temperature=0,seed=0,ignore_eos=True)
  req=urllib.request.Request(f'http://localhost:{port}/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  start=time.perf_counter()
  with urllib.request.urlopen(req,timeout=300) as f:r=json.load(f)
  responses.append(dict(c=con,i=i,wall_ms=(time.perf_counter()-start)*1000,response=r))
(root/f'fixed-prefill-{name}.json').write_text(json.dumps(responses,indent=2))
(root/(name+'.diagnostics.done')).touch()
