from previous.common import *
sys.path.insert(0,'/work/atom-actual-microbench/mhc')
from native_api import NativeMHC
weights=torch.load("/work/full-attention-native/prefix/mhc-fixture.pt",weights_only=True)
from previous.isolated import mhc_pre,mhc_mid,tail
from aiter.ops.triton.fusions.mhc import _gluon_post_pre_layouts,_gluon_reduce_apply_layouts
ll=_gluon_post_pre_layouts(4,16,64,32,2,True,4);vv=_gluon_reduce_apply_layouts(4,1,4,64)
w,s,b,an=[weights[k].cuda().contiguous() for k in ["HAF","HAS","HAB","AN"]];n=NativeMHC(w,s,b,an)
torch.manual_seed(42)
r=torch.randn((M,4,4096),device='cuda',dtype=torch.bfloat16)*.05;rs=aiter.mhc_res_shuffle(r);plain=torch.empty_like(r)
mix=torch.zeros((M,24),device='cuda');x=torch.empty((M,4096),device='cuda',dtype=torch.bfloat16)
e=lambda sh:torch.empty(sh,device='cuda');z=lambda:torch.zeros(1,device='cuda',dtype=torch.int32)
gm=e((64,M,32));sq=e((64,M));po=e((M,4,1));co=e((M,4,4));ct=z();h=z();args=(n.fn.view(torch.bfloat16),gm,sq,po,co,ct)
def fp():
 mhc_pre[(256,)](rs,plain,mix,x,s,b,an,args,h,M,ll,vv,**kw)
 return x
ref,post,comb=n(rs);fp();torch.cuda.synchronize()
bench('mhc_pre',{'native':lambda:n(rs),'fused':fp},dict(norm=err(x,ref),residual_exact=torch.equal(plain,r),post=err(mix[:,4:8],post),comb=err(mix[:,8:].reshape(M,4,4),comb),health=int(h)))
y=torch.randn_like(x)*.05;nr=torch.empty_like(r);nm=torch.empty_like(mix);ny=torch.empty_like(x)
def fpp():
 mhc_mid[(256,)](y,r,mix,nr,nm,ny,s,b,an,args,h,M,ll,vv,**kw)
 return ny
def npp():return n.post_pre(y,rs,post,comb)
rr=npp();fpp();torch.cuda.synchronize()
bench('mhc_post_pre',{'native':npp,'fused':fpp},dict(norm=err(ny,rr[2]),residual=err(nr,rr[3].reshape(64,4,M,64).permute(2,1,0,3).reshape(M,4,4096)),health=int(h)))
a=torch.randn_like(x);bb=torch.randn_like(x);summ=torch.empty_like(x);out=torch.empty_like(r);refout=torch.empty_like(r);counter=z()
def nt():
 aiter.mhc_post(refout,a+bb,rs,post.unsqueeze(-1),comb,res_preshuffle=True)
 return refout
def ft():
 tail[(256,)](r,a,bb,summ,mix,out,counter,h,M,**kw)
 return out
nt();ft();torch.cuda.synchronize()
# Native post returns plain residual; ATOM reshuffles before the next pre.
# Report both layout interpretations to verify actual API behavior.
check=dict(raw=err(out,refout),plain=err(out.reshape(64,4,M,64).permute(2,1,0,3).reshape(M,4,4096),refout),health=int(h))
bench('sum_final_mhc_post',{'native':nt,'fused':ft},check)
