from common import *
from fused import project_fused
from tdm_opt import project_tdm
from singlewave import project_singlewave
from initall import initialize_all
import os
out=root/'final-benchmark';out.mkdir(exist_ok=True)
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
from accepted import QAKVNorm
s16=QAKVNorm(split=16);s32=QAKVNorm(split=32)
runs=[('original_2launch',original),('original_fused_G',original_g),('optimized_2launch',two),('accepted_split16',lambda w:s16(X,XS,w,WS,NW,out=Y)),('accepted_split32',lambda w:s32(X,XS,w,WS,NW,out=Y))]
resources={}
for n,f in runs:
 initialize_all[(6,)](Acc,Count,**kw);obj=f(W);torch.cuda.synchronize()
 if n=='accepted_split16':obj=s16.kernel
 if n=='accepted_split32':obj=s32.kernel
 isa=obj.asm['amdgcn'];(out/(n+'.amdgcn')).write_text(isa)
 (out/(n+'.llir')).write_text(obj.asm['llir'])
 resources[n]=dict(vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared,sgpr_spills=re.findall(r'\.sgpr_spill_count:\s*(\d+)',isa),vgpr_spills=re.findall(r'\.vgpr_spill_count:\s*(\d+)',isa),private_segment=re.findall(r'\.amdhsa_private_segment_fixed_size (\d+)',isa))
(out/'resources.json').write_text(json.dumps(resources,indent=2))
results=[]
# Working-set sensitivity; identical weights at different addresses. No claim of forced cold cache.
for copies in [1,16,128]:
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
# Separate instrumented trace, not used as the headline timing.
for n,f in runs:
 initialize_all[(6,)](Acc,Count,**kw)
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):f(W)
 for _ in range(10):cg.replay()
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  for _ in range(30):cg.replay()
  torch.cuda.synchronize()
 prof.export_chrome_trace(str(out/(n+'-trace.json')))
manifest=dict(gpu=torch.cuda.get_device_name(0),properties=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,triton=__import__('triton').__version__,source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')},fixture_sha256=hashlib.sha256((root/'fixture.pt').read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('DONE',flush=True)
