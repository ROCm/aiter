import time,json,urllib.request,concurrent.futures
from pathlib import Path
r=Path('/work/rollout-c1248')
while not all((r/(n+'.diagnostics.done')).exists() for n in ['native','previous','selected','replica']):time.sleep(2)
def work(name,port):
 result=[]
 for c in [1,2,4,8]:
  (r/f'prefill-{name}.tag').write_text(f'quality-{name}-c{c}')
  def one(row):
   prompt=[100+(j+row*13)%1000 for j in range(1024)]
   payload=dict(model='/data/DeepSeek-V4-Flash',prompt=prompt,max_tokens=128,temperature=0,seed=0,ignore_eos=True)
   req=urllib.request.Request(f'http://localhost:{port}/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=300) as f:out=json.load(f)
   ch=out['choices'][0];ch['index']=row
   return ch
  with concurrent.futures.ThreadPoolExecutor(c) as pool:choices=list(pool.map(one,range(c)))
  d=dict(choices=choices)
  result.append(dict(c=c,response=d))
  (r/f'quality-{name}.json').write_text(json.dumps(result,indent=2))
 return name,result
with concurrent.futures.ThreadPoolExecutor(4) as ex:
 data=dict(ex.map(lambda p:work(*p),[('native',8260),('previous',8270),('selected',8280),('replica',8290)]))
checks={}
for name in ['native','selected','replica']:
 checks[name]=[]
 for a,b in zip(data['previous'],data[name]):
  aa=sorted(a['response']['choices'],key=lambda x:x['index']);bb=sorted(b['response']['choices'],key=lambda x:x['index'])
  checks[name].append(dict(c=a['c'],same_text=[x['text']==y['text'] for x,y in zip(aa,bb)],choices=len(bb),finish_reasons=[x['finish_reason'] for x in bb]))
(r/'QUALITY_COMPARISON.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
(r/'quality.done').touch()
