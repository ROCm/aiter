from common import *
import os,time,traceback
out=root/os.environ.get('OUT','sweep1');out.mkdir(exist_ok=True)
configs=[dict(BN=bn,SPLIT=sp,DIRECT=d,WAVES=nw,BK=bk,UNROLL=1,UPSCALE=True,REPLICATE=True) for d in [False,True] for nw in [1,2,4] for bn in [16,32,64] for sp in [4,8,16] for bk in [64,128]]
if os.environ.get('QUICK'):configs=[dict(BN=32,SPLIT=8,DIRECT=d,WAVES=4,BK=128,UNROLL=1,UPSCALE=True,REPLICATE=True) for d in [False,True]]
names=['baseline_A'];runs=[lambda: (project[(384,)](X,XS,W,WS,NW,P,Acc,SS,Count,Y,MODE=0,**kw),reduce_norm[(2,)](P,Acc,NW,Y,ATOMIC=False,**kw))[0]]
resources={};failures=[];checks=[]
# SPLIT16 requires more scratch.
P=torch.empty((32,1536),device='cuda',dtype=torch.float32)
for c in configs:
 name='_'.join(str(v) for v in c.values())
 def run(c=c):
  o=project_opt[(1536//c['BN']*c['SPLIT'],)](X,XS,W,WS,P,**c,num_warps=c['WAVES'],enable_fp_fusion=False)
  reduce_opt[(2,)](P,NW,Y,SPLIT=c['SPLIT'],**kw)
  return o
 names.append(name);runs.append(run)
good=[]
for name,run in zip(names,runs):
 try:
  obj=run();torch.cuda.synchronize()
  isa=obj.asm['amdgcn'];(out/(name+'.amdgcn')).write_text(isa)
  resources[name]=dict(vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared,wmma=isa.count('v_wmma'),barriers=isa.count('s_barrier_signal'),ds_load=len(re.findall(r'\bds_load_',isa)),ds_store=len(re.findall(r'\bds_store_',isa)))
  for case,(xp,xs) in enumerate(bank):
   X.copy_(xp);XS.copy_(xs);ref=reference();P.fill_(float('nan'));Y.fill_(float('nan'));run();m=metric(Y,ref)
   checks.append(dict(name=name,case=case,**m));assert m['finite'] and m['relative_rms']<.001,m
  good.append((name,run));print('COMPILED',name,resources[name],flush=True)
 except Exception as e:
  failures.append(dict(name=name,error=str(e)));print('FAILED',name,str(e),flush=True)
(out/'resources.json').write_text(json.dumps(resources,indent=2));(out/'failures.json').write_text(json.dumps(failures,indent=2));(out/'correctness.json').write_text(json.dumps(checks,indent=2))
X.copy_(bank[0][0]);XS.copy_(bank[0][1]);torch.cuda.synchronize()
graphs=[];repeat=100
for name,run in good:
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
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   start.record();cg.replay();end.record();end.synchronize();samples[name].append(start.elapsed_time(end)*1000/repeat)
 rows={n:dict(median_us=statistics.median(v),min_us=min(v),max_us=max(v),samples_us=v) for n,v in samples.items()}
 rounds.append(dict(round=rnd,graph_repeats=repeat,results=rows));print('ROUND',rnd,sorted([(n,r['median_us']) for n,r in rows.items()],key=lambda x:x[1]),flush=True)
(out/'timings.json').write_text(json.dumps(rounds,indent=2));print('DONE',flush=True)
