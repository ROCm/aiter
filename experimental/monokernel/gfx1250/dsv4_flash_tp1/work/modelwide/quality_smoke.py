import json,urllib.request,os
from pathlib import Path
prompts=['The capital of France is','2 + 2 =','Write a Python function that adds two numbers.\n','The opposite of hot is','A triangle has three']
results=[]
for prompt in prompts:
 row={'prompt':prompt}
 for name,port in [('baseline',8400),('fused',int(os.getenv('FUSED_PORT','8300')))]:
  body={'model':'/data/DeepSeek-V4-Flash','prompt':prompt,'max_tokens':48,'temperature':0,'seed':0,'logprobs':1}
  req=urllib.request.Request(f'http://localhost:{port}/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=180) as f:out=json.load(f)
  row[name]=out['choices'][0]
 row['text_equal']=row['baseline']['text']==row['fused']['text'];results.append(row)
 print(json.dumps(row),flush=True)
Path('/work/modelwide/quality-smoke'+os.getenv('QUALITY_TAG','')+'.json').write_text(json.dumps(results,indent=2))
