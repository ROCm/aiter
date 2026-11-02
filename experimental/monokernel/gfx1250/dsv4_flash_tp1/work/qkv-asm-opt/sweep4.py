from common import *
import os,time,traceback
out=root/os.environ.get('OUT','sweep1');out.mkdir(exist_ok=True)
from fused import project_fused
configs=[dict(BN=bn,SPLIT=sp,DIRECT=True,WAVES=nw,BK=128,UNROLL=1,UPSCALE=True,REPLICATE=True,CACHE='.ca',SCALAR=True,PACKED=False,MODE=m) for nw,bn in [(1,16),(1,32),(2,32),(2,64),(4,64)] for sp in [1,2,4,8,16] for m in [0,1]]
names=['baseline_A','baseline_G','optimized_2launch'];runs=[lambda: (project[(384,)](X,XS,W,WS,NW,P,Acc,SS,Count,Y,MODE=0,**kw),reduce_norm[(2,)](P,Acc,NW,Y,ATOMIC=False,**kw))[0], lambda:project[(384,)](X,XS,W,WS,NW,P,Acc,SS,Count,Y,MODE=6,**kw), lambda:(project_opt[(384,)](X,XS,W,WS,P,BN=32,SPLIT=8,DIRECT=True,WAVES=2,BK=128,UNROLL=1,UPSCALE=True,REPLICATE=True,CACHE='.ca',SCALAR=True,PACKED=False,num_warps=2,enable_fp_fusion=False),reduce_opt[(2,)](P,NW,Y,SPLIT=8,**kw))[0]]
resources={};failures=[];checks=[]
P=torch.empty((32,1536),device='cuda',dtype=torch.float32)
for c in configs:
 name='_'.join(str(v) for v in c.values())
 def run(c=c):
  return project_fused[(1536//c['BN']*c['SPLIT'],)](X,XS,W,WS,P,NW,Acc,Count,Y,**c,num_warps=c['WAVES'],enable_fp_fusion=False)
 names.append(name);runs.append(run)
good=[]
for name,run in zip(names,runs):
 try:
  initialize[(6,)](Acc,Count,**kw)
  obj=run();torch.cuda.synchronize()
  isa=obj.asm['amdgcn'];(out/(name+'.amdgcn')).write_text(isa)
  resources[name]=dict(vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared,wmma=isa.count('v_wmma'),barriers=isa.count('s_barrier_signal'),ds_load=len(re.findall(r'\bds_load_',isa)),ds_store=len(re.findall(r'\bds_store_',isa)))
  for case,(xp,xs) in enumerate(bank):
   initialize[(6,)](Acc,Count,**kw);X.copy_(xp);XS.copy_(xs);ref=reference();P.fill_(float('nan'));Y.fill_(float('nan'));run();m=metric(Y,ref)
   checks.append(dict(name=name,case=case,**m));assert m['finite'] and m['relative_rms']<.001,m
  good.append((name,run));print('COMPILED',name,resources[name],flush=True)
 except Exception as e:
  failures.append(dict(name=name,error=str(e)));print('FAILED',name,str(e),flush=True)
(out/'resources.json').write_text(json.dumps(resources,indent=2));(out/'failures.json').write_text(json.dumps(failures,indent=2));(out/'correctness.json').write_text(json.dumps(checks,indent=2))
X.copy_(bank[0][0]);XS.copy_(bank[0][1]);torch.cuda.synchronize()
graphs=[];repeat=100
for name,run in good:
 initialize[(6,)](Acc,Count,**kw)
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):
  for _ in range(repeat):run()
 for _ in range(5):cg.replay()
 graphs.append((name,cg))
torch.cuda.synchronize();rounds=[]
for rnd in [1,2]:
 samples={n:[] for n,_ in graphs}
 for sample in range(20):
  for name,cg in (graphs if sample%2==0 else reversed(graphs)):
   initialize[(6,)](Acc,Count,**kw)
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   start.record();cg.replay();end.record();end.synchronize();samples[name].append(start.elapsed_time(end)*1000/repeat)
 rows={n:dict(median_us=statistics.median(v),min_us=min(v),max_us=max(v),samples_us=v) for n,v in samples.items()}
 rounds.append(dict(round=rnd,graph_repeats=repeat,results=rows));print('ROUND',rnd,sorted([(n,r['median_us']) for n,r in rows.items()],key=lambda x:x[1])[:20],flush=True)
(out/'timings.json').write_text(json.dumps(rounds,indent=2));print('DONE',flush=True)
