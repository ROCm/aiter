import os,sys,json
# Use the same native backend default as ATOM serving.
sys.path[:0]=['/app/ATOM','/app/aiter',os.path.dirname(__file__)]
import numpy as np
import torch,triton
from compressor import compressor
from atom.model_ops.v4_kernels.fused_compress import fused_compress_attn
from atom.model_ops.v4_kernels.state_writes import update_compressor_states
from atom.model_ops.v4_kernels.compress_plan import CompressPlan
parallel=os.getenv("PARALLEL_POOL","0")=="1"
suffix="-parallel" if parallel else ""
case=int(sys.argv[1]);torch.manual_seed(730+case)
r=128 if case==1 else 4;d=128 if case==2 else 512;overlap=r==4
quant=case==2;df=d*(2 if overlap else 1);k=r*(2 if overlap else 1);state=k+(4 if case==3 else 0)
dev='cuda';rand=lambda *shape: torch.randn(*shape,device=dev)
kv=rand(1,df).bfloat16();score=rand(1,df).bfloat16();ape=rand(r,df);norm=rand(d)
cos=rand(2304,32).bfloat16();sin=rand(2304,32).bfloat16()
ks=rand(2,state,df);ss=rand(2,state,df);ks2=ks.clone();ss2=ss.clone()
cache=torch.zeros((16,256//r,d),device=dev,dtype=torch.float8_e4m3fn if quant else torch.bfloat16);cache2=cache.clone()
sc=torch.zeros((16,256//r),device=dev);sc2=sc.clone()
blocks=torch.arange(16,device=dev,dtype=torch.int32)[None,:]
si=torch.tensor([0],device=dev,dtype=torch.int32);so=torch.tensor([1 if case==3 else 0],device=dev,dtype=torch.int32)
cp=torch.zeros((1,4),device=dev,dtype=torch.int32);wp=cp.clone()
plan=CompressPlan(cp,wp,1,1,np.array([0,1],np.int32))
def native():
 fused_compress_attn(kv_in=kv,score_in=score,kv_state=ks,score_state=ss,plan=plan,state_slot_mapping=si,ape=ape,rms_weight=norm,rms_eps=1e-6,cos_cache=cos,sin_cache=sin,kv_cache=cache,block_tables=blocks,k_per_block=256//r,overlap=overlap,ratio=r,head_dim=d,rope_head_dim=64,quant=quant,cache_scale=sc,use_ue8m0=True,preshuffle=True,fp8_max=448.)
 update_compressor_states(kv,score,ape,ks,ss,write_plan=wp,state_slot_mapping=so,ratio=r,overlap=overlap)
def fused():
 return compressor[(1,)](kv,score,ape,norm,cos,sin,cp,wp,si,so,ks2,ss2,cache2,sc2,blocks,d,r,overlap,state,ks2.stride(0),ks2.stride(1),ss2.stride(0),ss2.stride(1),cache2.stride(0),cache2.stride(1),sc2.stride(0),quant,PARALLEL_POOL=parallel,num_warps=4)
results=[]
for pos in [r-1,1023,1024,1151,1535,2047]:
 cp.copy_(torch.tensor([[0,0,pos if (pos+1)%r==0 else -1,k-1]],device=dev,dtype=torch.int32));wp.copy_(torch.tensor([[0,0,pos,0]],device=dev,dtype=torch.int32))
 ks2.copy_(ks);ss2.copy_(ss);cache2.copy_(cache);sc2.copy_(sc)
 native();kernel=fused();torch.cuda.synchronize()
 diff=(cache.float()-cache2.float()).abs()
 results.append(dict(position=pos,cache_max_abs=diff.max().item(),cache_mismatch=(diff!=0).sum().item(),scale_equal=torch.equal(sc,sc2),kv_state_equal=torch.equal(ks,ks2),score_state_equal=torch.equal(ss,ss2)))
 print(results[-1],flush=True)
# Repeat graph replay to test stable state and benchmark second timing round.
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch
import statistics
graphs={name:FlatGraphBatch(fn,32) for name,fn in [('native',native),('fused',fused)]}
times=[]
for round in range(2):
 raw={name:[] for name in graphs}
 for gr in graphs.values():
  for _ in range(5):gr.replay()
 for sample in range(40):
  for name in list(graphs)[::1 if sample%2 else -1]:
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   a.record();graphs[name].replay();b.record();b.synchronize()
   raw[name].append(a.elapsed_time(b)*1000/32)
 times.append({name+'_us':statistics.median(v) for name,v in raw.items()})
report=dict(case=case,ratio=r,head_dim=d,quant=quant,validation=results,timings=times,registers=kernel.n_regs,shared=kernel.metadata.shared)
open(os.path.join(os.path.dirname(__file__),f'compressor-{case}{suffix}.json'),'w').write(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
