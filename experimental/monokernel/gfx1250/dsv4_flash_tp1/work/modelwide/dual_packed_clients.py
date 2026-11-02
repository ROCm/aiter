import concurrent.futures,json,urllib.request,time
from pathlib import Path
root=Path(__file__).parent
cases=[('The capital of France is',8),([42]*1024,8),([42]*1151,4)]
def run(item):
 mode,port=item
 for _ in range(180):
  try:
   urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2);break
  except Exception:time.sleep(2)
 results=[]
 for prompt,n in cases:
  req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=json.dumps(dict(model='/data/DeepSeek-V4-Flash',prompt=prompt,max_tokens=n,temperature=0,ignore_eos=True)).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=1200) as r:results.append(json.load(r))
  print(mode,len(results),flush=True)
  (root/f'dual-{mode}-responses.json').write_text(json.dumps(results,indent=2))
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 for result in pool.map(run,[('compressor-packed',10100)]):pass
