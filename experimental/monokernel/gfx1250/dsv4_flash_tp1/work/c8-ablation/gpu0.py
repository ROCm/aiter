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
bench('mhc_pre',{'native':lambda:n(rs),'fused':fused_mhc},err(v,ref))
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
bench('mhc_post',{'native':native_post,'fused':fused_post},dict(**err(out,refp),note='extracted post boundary; native res_preshuffle controls input layout'))
from output_defs import batched,scratch
sys.path.insert(0,'/work/full-attention-native/output-native')
from native_output import NativeOutput
no=NativeOutput();no.positions=torch.full((8,),1024,device='cuda',dtype=torch.int64)
f=no.fixture;wa=f['OA'].cuda().contiguous();sa=f['OAS'].cuda().view(torch.uint8).contiguous();wb=f['OB'].cuda().contiguous();sb=f['OBS'].cuda().view(torch.uint8).contiguous()
o=torch.randn((8,64,512),device='cuda',dtype=torch.bfloat16)*.1
sys.path.insert(0,'/work/atom-actual-microbench/qkv')
from native import NativeQKV
nq=NativeQKV();c=nq.cos[1024].reshape(1,32).repeat(8,1);sn=nq.sin[1024].reshape(1,32).repeat(8,1);a=scratch(8)
def fo():
 batched[(256,)](o,c,sn,wa,sa,wb,sb,*a,8,**kw)
 return a[7]
rr=no(o);vv=fo();torch.cuda.synchronize();bench('output_projection',{'native':lambda:no(o),'fused':fo},err(vv,rr))
