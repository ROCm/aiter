import json,urllib.request
payload=dict(model='/data/DeepSeek-V4-Flash',prompt='The capital of France is',max_tokens=8,temperature=0)
r=urllib.request.Request('http://localhost:8000/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(r,timeout=180) as response:print(response.read().decode())
