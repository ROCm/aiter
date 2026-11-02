import json,time,urllib.request
from pathlib import Path
out=Path('/work/latency-overlap')
base=Path('/work/concurrency-rollout')
for _ in range(900):
 if all((base/f'latency-overlap-{t}-c8-round2.json').exists() for t in ['native','previous','selected']):break
 time.sleep(1)
else:raise RuntimeError('Second client results did not complete')
prompts=['The capital of France is','Calculate 17 * 23. Show the calculation and give the answer.','Write a Python function that returns the sum of the squares of a list of numbers.']
results={}
for name,port in [('native',8230),('previous',8240),('selected',8250)]:
 results[name]=[]
 for p in prompts:
  payload=dict(model='/data/DeepSeek-V4-Flash',prompt=p,max_tokens=128,temperature=0,seed=0)
  req=urllib.request.Request(f'http://localhost:{port}/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=120) as f:response=json.load(f)
  results[name].append(dict(prompt=p,text=response['choices'][0]['text'],usage=response.get('usage'),finish_reason=response['choices'][0]['finish_reason']))
(out/'serving-text-check.json').write_text(json.dumps(results,indent=2))
print({name:[a['text']==b['text'] for a,b in zip(results['previous'],v)] for name,v in results.items()},flush=True)
