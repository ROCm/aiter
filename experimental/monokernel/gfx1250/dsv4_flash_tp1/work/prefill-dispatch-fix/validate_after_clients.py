from pathlib import Path
import json,time,subprocess,urllib.request,concurrent.futures
r=Path('/work/prefill-dispatch-fix')
for _ in range(1800):
 try:
  done=all(json.loads(Path('/work/concurrency-rollout',f'prefillfix-{tag}-c8-round2.json').read_text())['completed']==64 for tag in ['native','fused'])
  if done:break
 except (OSError,ValueError,KeyError):pass
 time.sleep(1)
else:raise RuntimeError('Performance clients incomplete')
def request(port,n):
 data=dict(model='/data/DeepSeek-V4-Flash',prompt=[100]*n,max_tokens=8,temperature=0,ignore_eos=True,seed=0)
 req=urllib.request.Request(f'http://localhost:{port}/v1/completions',json.dumps(data).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as q:return json.load(q)
checks=[]
for n in [1,2,4,8,1024]:
 with concurrent.futures.ThreadPoolExecutor(2) as e:
  a,b=list(e.map(lambda port:request(port,n),[12500,12800]))
 checks.append(dict(input_tokens=n,native=a,fixed=b,text_equal=a['choices'][0]['text']==b['choices'][0]['text']))
(r/'tiny-prefill-checks.json').write_text(json.dumps(checks,indent=2));print('TINY PREFILL',[(x['input_tokens'],x['text_equal']) for x in checks],flush=True)
subprocess.run(['bash','/work/batch-full/eval.sh','12800','prefillfix-fused-c8'],check=True)
