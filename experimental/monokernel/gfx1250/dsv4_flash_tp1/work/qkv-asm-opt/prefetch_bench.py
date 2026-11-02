from common import *
from prefetch import project_prefetch as project_fused
from tdm_opt import project_tdm
from singlewave import project_singlewave
from initall import initialize_all
import os
out=root/'prefetch-bench';out.mkdir(exist_ok=True)
Count=torch.empty(1536,device='cuda',dtype=torch.int32);P=torch.empty((32,1536),device='cuda')
cbase=dict(BN=128,SPLIT=16,DIRECT=True,WAVES=4,BK=128,UNROLL=1,UPSCALE=True,REPLICATE=True,CACHE='.ca',SCALAR=True,PACKED=False,MODE=1,OUTLAY=1,NVEC=1,COUNTSTRIDE=128,NATIVE=0)
def original(w):
 o=project[(384,)](X,XS,w,WS,NW,P,Acc,SS,Count,Y,MODE=0,**kw);reduce_norm[(2,)](P,Acc,NW,Y,ATOMIC=False,**kw);return o
def original_g(w):return project[(384,)](X,XS,w,WS,NW,P,Acc,SS,Count,Y,MODE=6,**kw)
def two(w):
 o=project_opt[(384,)](X,XS,w,WS,P,BN=32,SPLIT=8,DIRECT=True,WAVES=2,BK=128,UNROLL=1,UPSCALE=True,REPLICATE=True,CACHE='.ca',SCALAR=True,PACKED=False,num_warps=2,enable_fp_fusion=False);reduce_opt[(2,)](P,NW,Y,SPLIT=8,**kw);return o
def fused(w):return project_fused[(192,)](X,XS,w,WS,P,NW,Acc,Count,Y,**cbase,**kw)
def tdm(w):return project_tdm[(192,)](X,XS,w,WS,P,NW,Acc,Count,Y,**dict(cbase,UNROLL=2,OUTLAY=0),SHARED=0,**kw)
def single(w):return project_singlewave[(384,)](X,XS,w,WS,P,NW,Acc,Count,Y,**dict(cbase,SPLIT=32,NVEC=8),**kw)
def pf1(w):return project_fused[(192,)](X,XS,w,WS,P,NW,Acc,Count,Y,**cbase,PREFETCH=1,**kw)
def pf2(w):return project_fused[(192,)](X,XS,w,WS,P,NW,Acc,Count,Y,**cbase,PREFETCH=2,**kw)
runs=[('original_2launch',original),('optimized_2launch',two),('optimized_fused_BF16',fused),('TDM_L2_prefetch',pf1),('TDM_L2_speculative_prefetch',pf2)]
for n,f in runs:
 initialize_all[(6,)](Acc,Count,**kw);obj=f(W);torch.cuda.synchronize();(out/(n+'.amdgcn')).write_text(obj.asm['amdgcn'])
results=[]
# Working-set sensitivity; identical weights at different addresses. No claim of forced cold cache.
for copies in [1,128]:
 wb=W.view(torch.uint8)[None].repeat(copies,1,1).view(W.dtype)
 views=[wb[i] for i in range(copies)]
 graphs=[];repeat=128
 for n,f in runs:
  initialize_all[(6,)](Acc,Count,**kw)
  cg=torch.cuda.CUDAGraph()
  with torch.cuda.graph(cg):
   for i in range(repeat):f(views[i%copies])
  for _ in range(5):cg.replay()
  graphs.append((n,cg))
 torch.cuda.synchronize()
 for rnd in [1,2]:
  samples={n:[] for n,_ in runs}
  for sample in range(30):
   for n,cg in (graphs if sample%2==0 else reversed(graphs)):
    initialize_all[(6,)](Acc,Count,**kw)
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();cg.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/repeat)
  rows={n:dict(median_us=statistics.median(v),min_us=min(v),max_us=max(v),samples_us=v) for n,v in samples.items()}
  results.append(dict(copies=copies,weight_working_set_bytes=wb.numel()*wb.element_size(),round=rnd,repeats=repeat,results=rows));print(copies,rnd,{n:r['median_us'] for n,r in rows.items()},flush=True)
  (out/'timings.json').write_text(json.dumps(results,indent=2))
 del graphs,views,wb
print('DONE',flush=True)
