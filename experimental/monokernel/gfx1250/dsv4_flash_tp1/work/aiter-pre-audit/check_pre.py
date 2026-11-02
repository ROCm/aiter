import sys,json,torch,aiter,triton
sys.path[:0]=['/work/integrated-c8','/work/batch-full','/work/modelwide','/work/concurrency-rollout','/work/full-attention-native','/work/full-attention-native/prefix']
from mhc_pre_aiter import pre
from mhc_batch import mhc_batch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import sync
from aiter.ops.triton.fusions.mhc import _gluon_post_pre_layouts,_gluon_reduce_apply_layouts
@g.jit
def new(R,Mix,Y,S,B,NW,Args,H,C,M:gl.constexpr,L:gl.constexpr,V:gl.constexpr):
 pre(R,Mix,Y,S,B,NW,Args,H,M,L,V)
 sync(C,H,256,3)
@g.jit
def old(R,W,S,B,NW,Mix,Temp,Raw,Y,C,H,M:gl.constexpr):
 mhc_batch(R,W,S,B,NW,Mix,Temp,Raw,Y,C,H,M)
 sync(C,H,256,3)
def err(x,y):
 x=x.float();y=y.float();return dict(exact=bool(torch.equal(x,y)),max_abs=float((x-y).abs().max()),relative_rms=float(((x-y).square().mean()/y.square().mean().clamp_min(1.e-20)).sqrt()))
torch.manual_seed(42);results=[]
for m in [1,2,4,8]:
 R=torch.randn((m,4,4096),device='cuda',dtype=torch.bfloat16);W=torch.randn((24,16384),device='cuda')*.002
 S=torch.ones(3,device='cuda');B=torch.randn(24,device='cuda')*.1;NW=torch.ones(4096,device='cuda',dtype=torch.bfloat16)
 wp=aiter.mhc_shuffle_fn(W).view(torch.bfloat16)
 def zero(shape,dtype=torch.float32):return torch.zeros(shape,device='cuda',dtype=dtype)
 Args=(wp,zero((64,m,32)),zero((64,m)),zero((m,4,1)),zero((m,4,4)),zero(1,torch.int32))
 H=zero(1,torch.int32);C=zero(1,torch.int32);Mix=zero((m,24));Y=zero((m,4096),torch.bfloat16)
 OC=zero(1,torch.int32);OH=zero(1,torch.int32);OM=zero((m,24));OY=zero((m,4096),torch.bfloat16);Temp=zero((m,128,24));Raw=zero((m,4096),torch.bfloat16)
 L=_gluon_post_pre_layouts(4,16,64,32,2,True,4);V=_gluon_reduce_apply_layouts(4,1,4,64)
 nf=lambda:new[(256,)](R,Mix,Y,S,B,NW,Args,H,C,m,L,V,num_warps=4,enable_fp_fusion=False)
 of=lambda:old[(256,)](R,W,S,B,NW,OM,Temp,Raw,OY,OC,OH,m,num_warps=4,enable_fp_fusion=False)
 obj=nf();of();torch.cuda.synchronize()
 ref=aiter.mhc_pre(R,wp.view(torch.int32),S,B,norm_weight=NW,hc_post_mult_value=2.,w_preshuffle_bf16=1)
 torch.cuda.synchronize()
 row=dict(m=m,new_vs_native=dict(norm=err(Y,ref[2]),post=err(Args[3],ref[0]),comb=err(Args[4],ref[1])),old_vs_native=dict(norm=err(OY,ref[2]),post=err(OM[:,4:8],ref[0].reshape(m,4)),comb=err(OM[:,8:].reshape(m,4,4),ref[1])),new_vs_old=err(Y,OY),health=[int(H),int(OH)],vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills)
 assert row['health']==[0,0] and obj.n_spills==0 and torch.isfinite(Y).all()
 row['rounds']=[dict(new_us=triton.testing.do_bench_cudagraph(nf,rep=300)*1000,old_us=triton.testing.do_bench_cudagraph(of,rep=300)*1000) for _ in range(2)];row.update(row['rounds'][1])
 print(json.dumps(row),flush=True);results.append(row)
open('/work/aiter-pre-audit/pre-results.json','w').write(json.dumps(results,indent=2))
