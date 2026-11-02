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


from mhc_native_region import region
from aiter.ops.triton.fusions.mhc import _gluon_post_pre_layouts,_gluon_reduce_apply_layouts
ll=_gluon_post_pre_layouts(4,16,64,32,2,True,4);vv=_gluon_reduce_apply_layouts(4,1,4,64)
gm=torch.empty((64,8,32),device='cuda');sq=torch.empty((64,8),device='cuda');nr=torch.empty_like(rs)
po=torch.empty((8,4,1),device='cuda');co=torch.empty((8,4,4),device='cuda');yy=torch.empty_like(x)
def fpp():
 region[(256,)](y,rs,n.fn.view(torch.bfloat16),post,comb,s,b,an,gm,sq,nr,po,co,yy,ct,h,ll,vv,num_warps=4,enable_fp_fusion=True)
 return yy
def npp():return n.post_pre(y,rs,post,comb)
rr=npp();fpp();torch.cuda.synchronize();check=dict(norm=err(yy,rr[2]),post=err(po,rr[0]),comb=err(co,rr[1]),residual=err(nr,rr[3]),health=int(h))
bench('mhc_native_region',{'native':npp,'fused':fpp},check)
