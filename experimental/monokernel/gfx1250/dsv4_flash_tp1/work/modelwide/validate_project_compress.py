import sys,os,json
sys.path[:0]=['/app/ATOM','/app/aiter',os.path.dirname(__file__)]
# Reuse the same native compressor fixture without running its tests.
source=open('/work/modelwide/validate_compressor.py').read().split('results=[]')[0]
exec(compile(source,'compressor_fixture','exec'))
from project_compress import project_compress,project_body
from aiter.tuned_gemm import tgemm
no_fma=os.getenv('NO_FMA','0')=='1'
n=2*df
x=torch.randn(1,4096,device='cuda',dtype=torch.bfloat16)*.05
w=torch.randn(n,4096,device='cuda',dtype=torch.bfloat16)*.02
real_weights=os.getenv('REAL_WEIGHTS','0')=='1'
if real_weights:
 from safetensors import safe_open
 layer=[2,3,2,40][case];prefix=f'layers.{layer}.attn.'+('indexer.' if quant else '')+'compressor.'
 index=json.load(open('/data/DeepSeek-V4-Flash/model.safetensors.index.json'))['weight_map']
 def load(suffix):
  key=prefix+suffix
  with safe_open('/data/DeepSeek-V4-Flash/'+index[key],framework='pt') as fh:return fh.get_tensor(key).cuda().contiguous()
 w=torch.cat([load('wkv.weight'),load('wgate.weight')]);ape=load('ape');norm=load('norm.weight')
y=torch.empty(1,n,device='cuda',dtype=torch.bfloat16)
counter=torch.zeros(1,device='cuda',dtype=torch.uint32);health=torch.zeros(1,device='cuda',dtype=torch.int32)
C=(ape,norm,cos,sin,cp,wp,si,so,ks2,ss2,cache2,sc2,blocks)
meta=(d,r,overlap,state,ks2.stride(0),ks2.stride(1),ss2.stride(0),ss2.stride(1),cache2.stride(0),cache2.stride(1),sc2.stride(0),quant)
kv_orig=kv;score_orig=score
native_compress=native
def native():
 global kv,score
 out=tgemm.mm(x,w,None,otype=torch.bfloat16)
 kv=out[:,:df];score=out[:,df:]
 native_compress()
 return out

def fused():
 return project_compress[(256,)](x,w,y,C,counter,health,meta,n,num_warps=4,enable_fp_fusion=not no_fma)
checks=[]
for pos in [1023,1024,1151,1535,2047]:
 cp.copy_(torch.tensor([[0,0,pos if (pos+1)%r==0 else -1,k-1]],device=dev,dtype=torch.int32));wp.copy_(torch.tensor([[0,0,pos,0]],device=dev,dtype=torch.int32))
 ks2.copy_(ks);ss2.copy_(ss);cache2.copy_(cache);sc2.copy_(sc)
 ref=native();obj=fused();torch.cuda.synchronize()
 checks.append(dict(position=pos,projection_max_abs=(y.float()-ref.float()).abs().max().item(),projection_rms=(((y.float()-ref.float()).square().mean()/ref.float().square().mean()).sqrt()).item(),cache_max_abs=(cache.float()-cache2.float()).abs().max().item(),state_max_abs=(ks-ks2).abs().max().item(),health=health.item()))
 print(checks[-1],flush=True)
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch
import statistics
graphs={name:FlatGraphBatch(fn,16) for name,fn in [('native',native),('fused',fused)]};times=[]
for rnd in range(2):
 raw={name:[] for name in graphs}
 for gr in graphs.values():
  for _ in range(5):gr.replay()
 for sample in range(40):
  for name in list(graphs)[::1 if sample%2 else -1]:
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();graphs[name].replay();b.record();b.synchronize();raw[name].append(a.elapsed_time(b)*1000/16)
 times.append({name:statistics.median(v) for name,v in raw.items()})
report=dict(real_checkpoint_weights=real_weights,checks=checks,timings=times,vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills,health=health.item())
open(f'/work/modelwide/project-compress-{case}'+('-real' if real_weights else '')+('-nofma' if no_fma else '')+'.json','w').write(json.dumps(report,indent=2));open(f'/work/modelwide/project-compress-{case}'+('-real' if real_weights else '')+('-nofma' if no_fma else '')+'.amdgcn','w').write(obj.asm['amdgcn']);print(report,flush=True)
