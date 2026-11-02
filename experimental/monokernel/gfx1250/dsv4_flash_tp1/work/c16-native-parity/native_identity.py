from common import *
from identity_batch import identity_csa
from atom.model_ops.v4_kernels.csa_translate_pack import csa_translate_pack
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def run(Pos,Ptr,Blocks,Idx,Top,H,M:gl.constexpr):
 p=gl.program_id(0)
 if p<M*4:
  row=p//4
  identity_csa(Pos+row,Ptr+row,Blocks+row*16,Idx,Top+row*512,H,640,p%4)
for length in [256]:
 logits=torch.randn((M,1024),device='cuda');lens=torch.full((M,),length,device='cuda',dtype=torch.int32)
 top=torch.empty((M,512),device='cuda',dtype=torch.int32);out=torch.empty_like(top);pos=torch.full((M,),length*4-1,device='cuda',dtype=torch.int64)
 ptr=torch.arange(0,(M+1)*(length+128),length+128,device='cuda',dtype=torch.int32);blocks=torch.arange(M*16,device='cuda',dtype=torch.int32).reshape(M,16)
 idx=torch.full((M*(length+128),),-7,device='cuda',dtype=torch.int32);idx2=idx.clone();bid=torch.arange(M,device='cuda',dtype=torch.int32);h=torch.zeros(1,device='cuda',dtype=torch.int32)
 def native():
  aiter.top_k_per_row_decode(logits,1,lens,top,M,1024,1,k=512)
  csa_translate_pack(top,blocks,pos,ptr,bid,None,idx,envelope_rows=640,csa_block_capacity=64,window_size=128)
  return top
 def fused():
  run[(256,)](pos,ptr,blocks,idx2,out,h,M,**kw)
  return out
 native();fused();torch.cuda.synchronize()
 bench('identity_csa_'+str(length),{'native':native,'fused':fused},dict(topk_equal=torch.equal(top,out),indices_equal=torch.equal(idx,idx2),health=int(h)))
