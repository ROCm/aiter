from harness import *
from variants import CONFIGS,run,State
import os
out=root/'scratch-validation';out.mkdir(exist_ok=True)
inputs=[x.cuda() for x in f['inputs']]+[torch.zeros_like(X),torch.ones_like(X),-torch.ones_like(X)]
for seed in range(int(os.environ.get('RANDOM_CASES','256'))):
 gen=torch.Generator(device='cuda').manual_seed(32000+seed)
 x=torch.randn(4096,device='cuda',generator=gen)
 if seed%4==0:x=f['live'].cuda().float().reshape(-1)+x*.01
 else:x=x*2.**(seed%17-8)
 inputs.append(x.bfloat16())
names=['two_scratch_fp32', 'two_scratch_fp64', 'grid_scratch_fp32', 'grid_scratch_fp64', 'qready_scratch_fp32', 'qready_scratch_fp64', 'grid_scratch_rt_fp32']
states={n:State() for n in names};graphs={}
for n,s in states.items():
 run(s,n);torch.cuda.synchronize();cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):run(s,n)
 graphs[n]=cg
rows=[]
for i,x in enumerate(inputs):
 X.copy_(x)
 a=(i*137)*10000.**(-torch.arange(32,device='cuda',dtype=torch.float32)/32)
 Cos.copy_(a.cos());Sin.copy_(a.sin());r=reference(X)
 for rep in range(3):
  for n,s in states.items():
   graphs[n].replay()
   ms={k:metric(getattr(s,k),r[k]) for k in ['y','qp','q','kv']}
   row=dict(case=i,replay=rep,mode=n,metrics=ms,state=s.check_state())
   row['passed']=all(ms[k]['relative_rms']<.001 and ms[k]['finite'] for k in ['q','kv']) and all(v==0 if k=='health' else v for k,v in row['state'].items())
   rows.append(row)
 if i%8==0:print('case',i,'/',len(inputs),'failures',sum(not a['passed'] for a in rows),flush=True)
 (out/'results.json').write_text(json.dumps(rows,indent=2))
summary={n:dict(cases=len(inputs),outputs=sum(a['mode']==n for a in rows),failures=sum(a['mode']==n and not a['passed'] for a in rows),max_relative_rms={k:max(a['metrics'][k]['relative_rms'] for a in rows if a['mode']==n) for k in ['y','qp','q','kv']}) for n in names}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
