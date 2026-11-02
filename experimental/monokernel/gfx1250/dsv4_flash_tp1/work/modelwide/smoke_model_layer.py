import sys,os,json
sys.path[:0]=['/work/modelwide','/work/full-layer-native']
from layer_setup import LayerState
# Fixture setup adds older dependency paths at the front; restore this module first.
sys.path.insert(0,'/work/modelwide')
if os.getenv("MATCHED")=="1":
 from model_layer_matched import model_layer
elif os.getenv("SCALED_WMMA")=="1":
 from model_layer_scaled import model_layer
else:
 from model_layer import model_layer
import torch
case=int(sys.argv[1]);ratio=[0,4,128,4][case]
f=LayerState();s=f.a.s;a=s.attn
if os.getenv('MATCHED')=='1':
 a.m=torch.empty((64,64),device='cuda');a.l=torch.empty_like(a.m);a.a=torch.empty((64,64,512),device='cuda')
T=640 if ratio==4 else (144 if ratio==128 else 128)
a.cache=torch.zeros((T,512),device='cuda',dtype=torch.bfloat16)
a.indices=torch.zeros(T,device='cuda',dtype=torch.int32)
pos=torch.tensor([1023],device='cuda',dtype=torch.int64)
cos=f.a.nq.cos.reshape(-1,32).contiguous();sin=f.a.nq.sin.reshape(-1,32).contiguous()
real=torch.randn((2176,512),device='cuda',dtype=torch.bfloat16)*.02
blocks=torch.arange(16,device='cuda',dtype=torch.int32)
nc=1024//ratio if ratio else 0
idx=torch.cat((torch.arange(nc,device='cuda',dtype=torch.int32)//(256//max(ratio,1))*128+torch.arange(nc,device='cuda',dtype=torch.int32)%(256//max(ratio,1)),torch.arange(2048,2176,device='cuda',dtype=torch.int32)))
idx=torch.cat((idx,torch.zeros(640-len(idx),device='cuda',dtype=torch.int32)))
ptr=torch.tensor([0,nc+128],device='cuda',dtype=torch.int32);dest=torch.tensor([2175],device='cuda',dtype=torch.int32)
top=torch.empty(512,device='cuda',dtype=torch.int32);bias=torch.randn(256,device='cuda')*.01
counter=torch.zeros(1,device='cuda',dtype=torch.uint32)
def comp(d,r,quant):
 overlap=r==4;df=d*(2 if overlap else 1);k=r*(2 if overlap else 1)
 ape=torch.randn((r,df),device='cuda');norm=torch.ones(d,device='cuda',dtype=torch.bfloat16)
 plan=torch.tensor([[0,0,1023,k-1]],device='cuda',dtype=torch.int32);wp=plan.clone();slot=torch.zeros(1,device='cuda',dtype=torch.int32)
 ks=torch.randn((1,k,df),device='cuda');ss=torch.randn_like(ks)
 cache=torch.zeros((16,64,d),device='cuda',dtype=torch.float8_e4m3fn) if quant else real[:2048].view(16,128,512)[:,:256//r,:]
 scale=torch.zeros((16,256//r),device='cuda')
 C=(ape,norm,cos,sin,plan,wp,slot,slot,ks,ss,cache,scale,blocks)
 M=(d,r,overlap,k,ks.stride(0),ks.stride(1),ss.stride(0),ss.stride(1),cache.stride(0),cache.stride(1),scale.stride(0),quant)
 W=torch.randn((df*2,4096),device='cuda',dtype=torch.bfloat16)*.01;Y=torch.zeros((1,df*2),device='cuda',dtype=torch.bfloat16)
 return C,M,W,Y
mc,mm,mw,my=comp(512,ratio or 4,False);ic,im,iw,iy=comp(128,4,True)
def run():
 return model_layer[(256,)](f.attn_args(),f.f.args(),mc,ic,mw,iw,my,iy,pos,cos,sin,real,idx,ptr,dest,top,blocks,bias,mm,im,ratio,128,**f.f.meta(),HAS_BIAS=case==3,T=T,num_warps=4,enable_fp_fusion=False)
obj=run();torch.cuda.synchronize();out=f.f.out.clone();obj=run();torch.cuda.synchronize()
report=dict(case=case,ratio=ratio,finite=torch.isfinite(f.f.out).all().item(),health=f.health(),vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills,scope='compile and synchronization smoke test; mixed layer-zero and synthetic compressed weights, NOT model validation')
print(report,flush=True)
open(f'/work/modelwide/full-smoke-{case}'+os.getenv('TEST_TAG','')+('-scaled' if os.getenv('SCALED_WMMA')=='1' else '')+'.json','w').write(json.dumps(report,indent=2));open(f'/work/modelwide/full-smoke-{case}'+os.getenv('TEST_TAG','')+('-scaled' if os.getenv('SCALED_WMMA')=='1' else '')+'.amdgcn','w').write(obj.asm['amdgcn'])
