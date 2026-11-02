from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

@g.jit
def add(a,b):return a+b

@g.jit
def route_body(Logits,Hist,Offs,Pad,Map,Gather,Scatter,Gamma,Bias,HAS_BIAS:gl.constexpr):
 if gl.program_id(0)==0:
  L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0]);i=gl.arange(0,256,layout=L)
  x=gl.load(Logits+i).to(gl.float32)
  scores=gl.sqrt(gl.maximum(x,0.)+gl.log(1.+gl.exp(-gl.abs(x)))).to(gl.bfloat16).to(gl.float32)
  if HAS_BIAS:
   bias=gl.load(Bias+i)
   # AITER casts correction bias to BF16 before the ranking addition.
   scores=(scores+bias.to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16).to(gl.float32)
  remain=scores;chosen=gl.full((256,),0,gl.int32,L)
  for j in gl.static_range(6):
   best=gl.max(remain,0);idx=gl.max(gl.where(remain==best,i,-1),0)
   chosen=gl.where(i==idx,1,chosen);remain=gl.where(i==idx,float('-inf'),remain)
  if HAS_BIAS:
   scores=(scores-bias).to(gl.bfloat16).to(gl.float32)
  rank=gl.associative_scan(chosen,0,add)-chosen
  total=gl.sum(gl.where(chosen!=0,scores,0.),0)
  gamma=(scores/(total+1.e-20)*1.5).to(gl.bfloat16)
  gl.store(Hist+i,chosen);gl.store(Offs+i,rank)
  gl.store(Map+rank,i,chosen!=0);gl.store(Gamma+rank,gamma,chosen!=0)
  gl.store(Gather+rank,rank,chosen!=0);gl.store(Scatter+rank,rank,chosen!=0)
  gl.store(Pad,6)
