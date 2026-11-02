import json,urllib.request,concurrent.futures
from pathlib import Path
r=Path('/work/rollout-c1248')
def work(item):
 name,port=item;data=[]
 for c in [1,2,4,8]:
  (r/f'prefill-{name}.tag').write_text(f'controlled-quality-{name}-c{c}')
  payload=dict(model='/data/DeepSeek-V4-Flash',prompt=[100+j%1000 for j in range(1024)],max_tokens=128,n=c,temperature=0.00001,top_k=1,ignore_eos=True,return_token_ids=True)
  req=urllib.request.Request(f'http://localhost:{port}/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=300) as f:out=json.load(f)
  assert len(out['choices'])==c
  data.append(dict(c=c,response=out));(r/f'controlled-quality-{name}.json').write_text(json.dumps(data,indent=2))
 return name,data
with concurrent.futures.ThreadPoolExecutor(4) as ex:data=dict(ex.map(work,[('native',8260),('previous',8270),('selected',8280),('replica',8290)]))
checks={}
for n in ['native','selected','replica']:
 checks[n]=[]
 for a,b in zip(data['previous'],data[n]):
  aa=sorted(a['response']['choices'],key=lambda x:x['index']);bb=sorted(b['response']['choices'],key=lambda x:x['index'])
  checks[n].append(dict(c=a['c'],same_text=[x['text']==y['text'] for x,y in zip(aa,bb)]))
(r/'CONTROLLED_QUALITY_COMPARISON.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
