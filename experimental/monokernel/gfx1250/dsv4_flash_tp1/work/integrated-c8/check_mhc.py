import sys,json,torch
sys.path[:0]=['/work/batch-full','/work/modelwide','/work/full-attention-native/prefix']
from mhc_batch import mhc_batch
from mhc_precise import candidate_body
from post_batch import post_batch
from post_precise import post_body
from triton.experimental import gluon as g
@g.jit
def c1(R,W,S,B,AN,Mix,Temp,Raw,X,C,H,F):
 candidate_body(R,W,S,B,AN,Mix,Temp,Raw,X,C,H,FP32=F)
@g.jit
def post_c1(R,X,M,O,F):post_body(R,X,M,O,FP32=F)
torch.manual_seed(42)
res=[]
for m in [1,2,4,8]:
 d='cuda';r=torch.randn((m,4,4096),device=d,dtype=torch.bfloat16);f=r.float()+torch.randn((m,4,4096),device=d)*.0001
 w=torch.randn((24,16384),device=d,dtype=torch.float32)*.01;s=torch.ones(3,device=d);b=torch.randn(24,device=d)*.1;an=torch.randn(4096,device=d,dtype=torch.bfloat16)*.01+1
 def scratch(n):return [torch.zeros((n,24),device=d),torch.zeros((n,128,24),device=d),torch.zeros((n,4096),device=d,dtype=torch.bfloat16),torch.zeros((n,4096),device=d,dtype=torch.bfloat16),torch.zeros(1,device=d,dtype=torch.int32),torch.zeros(1,device=d,dtype=torch.int32)]
 a=scratch(m);ref=scratch(m)
 obj=mhc_batch[(256,)](r,w,s,b,an,*a,m,FP32=f,num_warps=4,enable_fp_fusion=False)
 for row in range(m):c1[(256,)](r[row],w,s,b,an,ref[0][row],ref[1][row],ref[2][row],ref[3][row],ref[4],ref[5],f[row],num_warps=4,enable_fp_fusion=False)
 torch.cuda.synchronize()
 checks={k:bool(torch.equal(x,y)) for k,x,y in zip(['mix','partials','raw','norm'],a[:4],ref[:4])}
 out=torch.empty_like(r);out32=torch.empty_like(f);rr=torch.empty_like(r);rr32=torch.empty_like(f)
 post_batch[(256,)](r,a[3],a[0],out,m,FP32=out32,num_warps=4,enable_fp_fusion=False)
 for row in range(m):post_c1[(256,)](r[row],a[3][row],a[0][row],rr[row],rr32[row],num_warps=4,enable_fp_fusion=False)
 torch.cuda.synchronize();checks.update(post=bool(torch.equal(out,rr)),post32=bool(torch.equal(out32,rr32)))
 row=dict(m=m,checks=checks,health=int(a[-1].item()),vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared);res.append(row);print(row,flush=True)
 assert all(checks.values()) and row['health']==0
open('/work/batch-full/mhc-checks.json','w').write(json.dumps(res,indent=2))
