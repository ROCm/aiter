import json,time,urllib.request,statistics,sys
from pathlib import Path
r=Path('/work/rollout-c1248');name,port,rnd=sys.argv[1:]
(r/f'prefill-{name}.tag').write_text(f'bulk-{name}-c8-round{rnd}')
times=[]
for i in range(8):
 payload=dict(model='/data/DeepSeek-V4-Flash',prompt=[100+j%1000 for j in range(1024)],n=8,max_tokens=1,temperature=0.00001,top_k=1,ignore_eos=True)
 req=urllib.request.Request(f'http://localhost:{port}/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 t=time.perf_counter()
 with urllib.request.urlopen(req,timeout=120) as f:d=json.load(f)
 times.append((time.perf_counter()-t)*1000);assert len(d['choices'])==8
out=dict(configuration=name,round=int(rnd),requests=8,completions_per_request=8,input_tokens_per_completion=1024,output_tokens_per_completion=1,median_response_ms=statistics.median(times),samples_ms=times)
(r/f'bulk-{name}-c8-round{rnd}.json').write_text(json.dumps(out,indent=2));print(out)
