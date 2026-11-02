import json,sys,urllib.request
prompts=['The capital of France is','Calculate 17 times 23.','Write a Python function that adds two numbers.','Explain why the sky appears blue in one sentence.']
results=[]
for prompt in prompts:
 data=dict(model='/data/DeepSeek-V4-Pro',prompt=prompt,max_tokens=64,temperature=0,seed=0)
 req=urllib.request.Request('http://localhost:8000/v1/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as f:results.append(dict(prompt=prompt,response=json.load(f)))
with open('/work/results/'+sys.argv[1]+'_completions.json','w') as f:json.dump(results,f,indent=2)
