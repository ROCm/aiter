from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

@g.jit
def identity_csa(Positions,Indptr,Blocks,Indices,Topk,Health,ENVELOPE:gl.constexpr):
 p=gl.program_id(0)
 if p<4:
  pos=gl.load(Positions);length=(pos+1)//4
  if length<=512:
   i=p*128+gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
   n=gl.load(Indptr+1)-gl.load(Indptr)-gl.minimum(pos+1,128)
   gl.store(Topk+i,gl.where(i<length,i,-1))
   block=gl.load(Blocks+i//64,i<n,0)
   gl.store(Indices+gl.load(Indptr)+i,block*ENVELOPE+i%64,i<n)
  elif p==0:
   gl.atomic_or(Health,4,sem='relaxed',scope='gpu')
