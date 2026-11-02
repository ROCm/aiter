import urllib.request,json,concurrent.futures
from pathlib import Path
r=Path('/work/prefill-dispatch-fix')
def request(port,n,k):
 data=dict(model='/data/DeepSeek-V4-Flash',prompt=[100]*n,max_tokens=k,temperature=0,ignore_eos=True,seed=0,logprobs=1)
 req=urllib.request.Request(f'http://localhost:{port}/v1/completions',json.dumps(data).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as q:return json.load(q)
results=[]
for n in [1,2,4,8,1024]:
 with concurrent.futures.ThreadPoolExecutor(2) as e:a,b=list(e.map(lambda p:request(p,n,1),[12500,12800]))
 results.append(dict(input_tokens=n,native=a,fixed=b,first_token_equal=a['choices'][0]['text']==b['choices'][0]['text']))
(r/'first-token-checks.json').write_text(json.dumps(results,indent=2));print([(d['input_tokens'],d['first_token_equal']) for d in results])
