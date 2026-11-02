from common import *
from tail_kernel import combine
from layout_batch import unshuffle,shuffle
from triton.experimental import gluon as g
@g.jit
def layouts(X,Y,Z):
 unshuffle(X,Y,8)
@g.jit
def shuf(X,Y):shuffle(X,Y,8)
torch.manual_seed(32)
r=torch.randn((8,4,4096),device='cuda',dtype=torch.bfloat16);rs=aiter.mhc_res_shuffle(r)
a=torch.randn((8,4096),device='cuda',dtype=torch.bfloat16);b=torch.randn_like(a);total=torch.empty_like(a)
p=torch.rand((8,4,1),device='cuda');c=torch.rand((8,4,4),device='cuda');mix=torch.zeros((8,24),device='cuda');mix[:,4:8]=p[:,:,0];mix[:,8:]=c.reshape(8,16)
out=torch.empty_like(r);ref=torch.empty_like(r);fp32=torch.empty_like(r,dtype=torch.float32);ct=torch.zeros(1,device='cuda',dtype=torch.int32);h=torch.zeros_like(ct)
def native():
 aiter.mhc_post(ref,a+b,rs,p,c,res_preshuffle=True)
 return ref
def fused():
 combine[(256,)](r,a,b,total,mix,out,ct,h,fp32,**kw)
 return out
native();fused();torch.cuda.synchronize()
ref_plain=ref
bench('combine_post',{'native':native,'fused':fused},err(out,ref_plain))
bench('residual_unshuffle',{'fused':lambda:layouts[(256,)](rs,out,fp32,**kw)},dict(note='extra monokernel layout conversion; native has no corresponding standalone operation'))
bench('residual_shuffle',{'fused':lambda:shuf[(256,)](r,out,**kw)},dict(note='monokernel residual layout conversion; native boundary layout handling differs'))
