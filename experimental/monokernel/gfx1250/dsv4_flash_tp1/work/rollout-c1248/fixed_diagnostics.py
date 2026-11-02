import json,time,urllib.request,sys
from pathlib import Path
root=Path('/work/rollout-c1248');name,port=sys.argv[1:]
responses=[]
for con in [1,8]:
 for i in range(5):
  (root/f'prefill-{name}.tag').write_text(f'fixed-prefill-{name}-c{con}-{i}')
  if i==3:(root/f'prefill-{name}.trace').touch()
  tokens=[100+j%1000 for j in range(1024)]
  payload=dict(model='/data/DeepSeek-V4-Flash',prompt=tokens,n=con,max_tokens=1,temperature=0.00001,seed=0,ignore_eos=True)
  req=urllib.request.Request(f'http://localhost:{port}/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  start=time.perf_counter()
  with urllib.request.urlopen(req,timeout=300) as f:r=json.load(f)
  assert len(r['choices'])==con,(con,r)
  responses.append(dict(c=con,i=i,wall_ms=(time.perf_counter()-start)*1000,response=r))
  (root/f'fixed-prefill-{name}.json').write_text(json.dumps(responses,indent=2))
(root/f'fixed-prefill-{name}.json').write_text(json.dumps(responses,indent=2))
(root/(name+'.diagnostics.done')).touch()
