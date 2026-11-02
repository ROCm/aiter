import urllib.request,json,time,concurrent.futures,pathlib
from transformers import AutoTokenizer
model='/data/DeepSeek-V4-Flash';tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True)
root=pathlib.Path('/work/batch-full');base_result=pathlib.Path('/work/concurrency-rollout/batch-full-native-c2-round2.json')
# Never contaminate an in-flight performance client.
for _ in range(900):
 if base_result.exists():
  try:
   with urllib.request.urlopen('http://localhost:11600/health',timeout=2) as r:
    if r.status==200:break
  except Exception:pass
 time.sleep(1)
else:raise RuntimeError('Readiness timeout')
context=tok.encode('This is background context about everyday arithmetic and counting. '*400,add_special_tokens=False)
def one(port,i):
 tail=tok.encode(f'\nQuestion: What is {17+i} plus {25+i}? Answer with the number.\nAnswer:',add_special_tokens=False)
 prompt=context[:1024-len(tail)]+tail
 body=dict(model=model,prompt=prompt,max_tokens=32,temperature=0,ignore_eos=True,seed=0,logprobs=1)
 req=urllib.request.Request(f'http://localhost:{port}/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)

references=[one(11600,i) for i in range(8)]
results=[]
for m in [2,4,8]:
 with concurrent.futures.ThreadPoolExecutor(max_workers=m) as pool:fused=list(pool.map(lambda i:one(11600,i),range(m)))
 match=[a['choices'][0]['text']==b['choices'][0]['text'] for a,b in zip(references,fused)]
 row=dict(concurrency=m,text_equal=match,reference=references[:m],fused=fused);results.append(row);print(dict(concurrency=m,text_equal=match),flush=True)
 (root/'batch-self-probe.json').write_text(json.dumps(results,indent=2))
