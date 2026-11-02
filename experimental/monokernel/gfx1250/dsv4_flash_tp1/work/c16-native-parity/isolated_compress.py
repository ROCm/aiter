from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from project_batch import project_body
from optimized_compressor import compressor_body,pool_partial,HCA_CONCURRENT,HCA_WIDTH,HCA_READY,hca_ready,CSA_POOL
from combined_scaled import sync
@g.jit
def batch(X,W,Y,C,COUNTER,HEALTH,META:gl.constexpr,N:gl.constexpr,M:gl.constexpr,Work=None):
 p=gl.program_id(0)
 project_body(X,W,Y,N,M)
 if Work is not None and HCA_CONCURRENT:
  pool_partial(C[8],C[9],C[4],C[6],Work,META[4],META[5],META[6],META[7],M,4,Task=p-N//16)
  if HCA_READY:hca_ready(COUNTER,HEALTH,M)
  elif p<N//16+M*(512//HCA_WIDTH):sync(COUNTER,HEALTH,N//16+M*(512//HCA_WIDTH),3)
 else:
  if p<N//16:sync(COUNTER,HEALTH,N//16,3)
 if Work is not None and not HCA_CONCURRENT:
  pool_partial(C[8],C[9],C[4],C[6],Work,META[4],META[5],META[6],META[7],M,4)
  sync(COUNTER+1,HEALTH,256,3)
 if p<M:
  CR=C[:4]+(C[4]+p*4,C[5]+p*4,C[6]+p,C[7]+p)+C[8:12]+(C[12]+p*16,)
  compressor_body(Y+p*N,Y+p*N+N//2,*CR,*META,OWNER=p,PARALLEL_POOL=CSA_POOL,Partial=None if Work is None else Work+p*6144,EARLY=HCA_CONCURRENT and Work is not None)
