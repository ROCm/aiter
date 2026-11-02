import concurrent.futures,json,sys,urllib.request
from pathlib import Path
mode=sys.argv[1];out={}
for c in (1,8,16,32,64,128):
 def request(i):
  prompt=('This is a simple arithmetic and reasoning exercise. '*80)+f'\nQuestion {i%8}: What is {17+i%8} times 23? Explain the calculation.\nAnswer:'
  data=dict(model='/data/DeepSeek-V4-Flash',prompt=prompt,max_tokens=16,temperature=0,seed=0,ignore_eos=True)
  req=urllib.request.Request('http://localhost:8000/v1/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=240) as f:response=json.load(f)
  return dict(index=i,prompt=prompt,text=response['choices'][0]['text'],usage=response.get('usage'))
 with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:out[str(c)]=list(ex.map(request,range(c)))
Path('/work/flash/'+mode+'_batch_completions.json').write_text(json.dumps(out,indent=2))
