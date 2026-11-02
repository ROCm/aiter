import urllib.request,json,time,concurrent.futures
from pathlib import Path
root=Path(__file__).parent
for _ in range(300):
 try:urllib.request.urlopen('http://localhost:10100/health',timeout=2);break
 except Exception:time.sleep(2)
def request(item):
 port,prompt=item
 req=urllib.request.Request(f'http://localhost:{port}/v1/completions',data=json.dumps(dict(model='/data/DeepSeek-V4-Flash',prompt=prompt,max_tokens=16,temperature=0,ignore_eos=True)).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=1200) as f:return json.load(f)
rows=[]
for c in [2,4,8]:
 prompts=[[42+i]*1024 for i in range(c)]
 with concurrent.futures.ThreadPoolExecutor(c) as pool:out=list(pool.map(request,[(10100,p) for p in prompts]))
 with concurrent.futures.ThreadPoolExecutor(c) as pool:ref=list(pool.map(request,[(10200,p) for p in prompts]))
 rows.append(dict(concurrency=c,candidate=out,native=ref,exact=[a['choices']==b['choices'] for a,b in zip(out,ref)]))
 (root/'qa-shadow-native-rms-quality.json').write_text(json.dumps(rows,indent=2));print(c,rows[-1]['exact'],flush=True)
