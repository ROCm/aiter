import sys,torch,json
sys.path[:0]=['/work/batch-full','/work/modelwide','/work/full-attention-native/prefix']
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from project_batch import project_body
from project_compress import project_body as single_project
from compressor_batch import compressor_body
from compressor import compressor_body as single_compress
from combined import sync
@g.jit
def batch(X,W,Y,C,COUNTER,HEALTH,META:gl.constexpr,N:gl.constexpr,M:gl.constexpr):
 p=gl.program_id(0)
 project_body(X,W,Y,N,M)
 sync(COUNTER,HEALTH,256,3)
 if p<M:
  CR=C[:4]+(C[4]+p*4,C[5]+p*4,C[6]+p,C[7]+p)+C[8:12]+(C[12]+p*16,)
  compressor_body(Y+p*N,Y+p*N+N//2,*CR,*META,OWNER=p)
@g.jit
def single(X,W,Y,C,COUNTER,HEALTH,META:gl.constexpr,N:gl.constexpr):
 single_project(X,W,Y,N)
 sync(COUNTER,HEALTH,256,3)
 single_compress(Y,Y+N//2,*C,*META)
def rand(*s):return torch.randn(*s,device='cuda')
def iz(*s):return torch.zeros(*s,device='cuda',dtype=torch.int32)
results=[]
for case in range(4):
 r=128 if case==1 else 4;d=128 if case==2 else 512;quant=case==2;df=d*(2 if r==4 else 1);state=(r*2 if r==4 else r)+(4 if case==3 else 0);n=df*2
 torch.manual_seed(713+case);w=rand(n,4096).bfloat16()*.02;ape=rand(r,df);norm=rand(d);cos=rand(2304,32).bfloat16();sin=rand(2304,32).bfloat16()
 for m in [1,2,4,8]:
  x=rand(m,4096).bfloat16()*.05;y=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
  ks=rand(m*2,state,df);ss=rand(m*2,state,df);cache=torch.zeros((m*16,256//r,d),device='cuda',dtype=torch.float8_e4m3fn if quant else torch.bfloat16);scale=torch.zeros((m*16,256//r),device='cuda')
  ksref=ks.clone();ssref=ss.clone();cref=cache.clone();sref=scale.clone()
  slots=torch.arange(m,device='cuda',dtype=torch.int32);outslots=slots+m if case==3 else slots
  blocks=torch.arange(m*16,device='cuda',dtype=torch.int32).reshape(m,16)
  pos=torch.tensor([1023,1024,1151,1535,2047,1027,1028,1407][:m],device='cuda',dtype=torch.int32)
  cp=iz(m,4);wp=iz(m,4);cp[:,2]=torch.where((pos+1)%r==0,pos,-1);cp[:,3]=r*(2 if r==4 else 1)-1;wp[:,2]=pos
  c=(ape,norm,cos,sin,cp,wp,slots,outslots,ks,ss,cache,scale,blocks)
  meta=(d,r,r==4,state,ks.stride(0),ks.stride(1),ss.stride(0),ss.stride(1),cache.stride(0),cache.stride(1),scale.stride(0),quant)
  counter=iz(1);health=iz(1)
  obj=batch[(256,)](x,w,y,c,counter,health,meta,n,m,num_warps=4,enable_fp_fusion=False)
  ys=[]
  for row in range(m):
   cr=(ape,norm,cos,sin,cp[row],wp[row],slots[row:row+1],outslots[row:row+1],ksref,ssref,cref,sref,blocks[row]);yr=torch.empty((1,n),device='cuda',dtype=torch.bfloat16)
   single[(256,)](x[row],w,yr,cr,iz(1),iz(1),meta,n,num_warps=4,enable_fp_fusion=False);ys.append(yr)
  torch.cuda.synchronize();checks=dict(projection=torch.equal(y,torch.cat(ys)),kv_state=torch.equal(ks,ksref),score_state=torch.equal(ss,ssref),cache=torch.equal(cache.view(torch.uint8),cref.view(torch.uint8)),scale=torch.equal(scale,sref))
  result=dict(case=case,m=m,checks=checks,health=int(health),vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills);print(result,flush=True);results.append(result);assert all(checks.values()) and result['health']==0
open('/work/batch-full/compressor-checks.json','w').write(json.dumps(results,indent=2))
