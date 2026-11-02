"""Second-client correctness smoke; latency here is not a serving benchmark."""
import concurrent.futures
import json
import os
import sys
import urllib.request
from pathlib import Path

model, mode = sys.argv[1:3]
root=Path('/work/gfx1250-integration')
for run in (1,2):
 out={}
 for c in (1,8,16,32,64,128):
  def request(i):
   prompt=('This is a simple arithmetic and reasoning exercise. '*24)+f'\nWhat is {17+i%8} times 23? Explain.\nAnswer:'
   data=dict(model='/data/DeepSeek-V4-'+model,prompt=prompt,max_tokens=24,temperature=0,seed=0,ignore_eos=True)
   req=urllib.request.Request('http://127.0.0.1:'+os.getenv('PORT','8000')+'/v1/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=240) as f:response=json.load(f)
   return dict(index=i,text=response['choices'][0]['text'],usage=response.get('usage'))
  with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:out[str(c)]=list(ex.map(request,range(c)))
  print(mode,'client',run,'C',c,'completed',len(out[str(c)]),flush=True)
 (root/f'{mode}_client{run}.json').write_text(json.dumps(out,indent=2)+'\n')
