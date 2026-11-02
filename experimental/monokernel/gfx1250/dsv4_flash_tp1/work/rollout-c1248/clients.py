import urllib.request,time,subprocess,sys,os,json
from pathlib import Path
name,port=sys.argv[1:];root=Path('/work/rollout-c1248')
for _ in range(2400):
 try:
  with urllib.request.urlopen(f'http://localhost:{port}/health',timeout=2) as r:
   if r.status==200:break
 except Exception:pass
 time.sleep(1)
else:raise RuntimeError('Server readiness timeout')
# All machines finish startup before measured traffic, avoiding shared CPU/load contention.
(root/(name+'.ready')).touch()
while not all((root/(n+'.ready')).exists() for n in ['native','previous','selected','replica']):time.sleep(1)
for con in [1,2,4,8]:
 for rnd in [1,2]:
  tag=f'{name}-c{con}-round{rnd}'
  (root/f'prefill-{name}.tag').write_text(tag)
  cmd=[sys.executable,'-m','atom.benchmarks.benchmark_serving','--model=/data/DeepSeek-V4-Flash','--backend=vllm',f'--base-url=http://localhost:{port}','--dataset-name=random','--random-input-len=1024','--random-output-len=1024','--random-range-ratio=1',f'--num-prompts={con*8}',f'--max-concurrency={con}','--request-rate=inf','--ignore-eos','--seed=0','--percentile-metrics=ttft,tpot,itl,e2el','--save-result','--save-detailed',f'--result-dir={root}',f'--result-filename={tag}.json']
  (root/(tag+'.start.json')).write_text(json.dumps(dict(time=time.time(),cmd=cmd)))
  with open(root/(tag+'.log'),'w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',tag,flush=True)
(root/(name+'.clients.done')).touch()
