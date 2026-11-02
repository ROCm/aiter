from common import *
from mhc_defs import mhc_batch,post_batch
sys.path.insert(0,'/work/atom-actual-microbench/mhc')
from native_api import NativeMHC
f=torch.load('/work/mhc-optimized/fixture.pt',weights_only=False)
print('fixture keys',list(f),flush=True)
# Reuse checkpoint mHC weights from the saved native fixture.
import setup as ps
w,s,b,an=ps.HAF,ps.HAS,ps.HAB,ps.AN
n=NativeMHC(w,s,b,an)
torch.manual_seed(42);r=torch.randn((8,4,4096),device='cuda',dtype=torch.bfloat16)*.05;rs=aiter.mhc_res_shuffle(r)
mix=torch.zeros((8,24),device='cuda');tmp=torch.zeros((8,128,24),device='cuda');raw=torch.empty((8,4096),device='cuda',dtype=torch.bfloat16);x=torch.empty_like(raw);ct=torch.zeros(1,device='cuda',dtype=torch.int32);h=torch.zeros_like(ct)
def fused_mhc():
 mhc_batch[(256,)](r,w,s,b,an,mix,tmp,raw,x,ct,h,8,**kw)
 return x
ref,post,comb=n(rs);v=fused_mhc();torch.cuda.synchronize()

y=torch.randn_like(x)*.05;out=torch.empty_like(r);out32=torch.empty_like(r,dtype=torch.float32)
def fused_post():
 post_batch[(256,)](r,y,mix,out,8,False,out32,**kw)
 return out
# Native extracted post; serving commonly fuses this with following pre.
post_ref=torch.empty_like(r)
def native_post():
 aiter.mhc_post(post_ref,y,rs,post.unsqueeze(-1),comb,res_preshuffle=True)
 return post_ref
fused_post();refp=native_post();torch.cuda.synchronize()


from tail_kernel import post_pre
mix2=torch.empty_like(mix);x2=torch.empty_like(x)
def fpp():
 post_pre[(256,)](r,y,mix,out,out32,w,s,b,an,mix2,tmp,raw,x2,ct,h,**kw)
 return x2
def npp():return n.post_pre(y,rs,post,comb)
rr=npp();vv=fpp();torch.cuda.synchronize()
print('post_pre outputs',[(v.shape,v.dtype) for v in rr],flush=True)
bench('mhc_post_pre',{'native':npp,'fused':fpp},err(vv,rr[2]))
