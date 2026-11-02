import sys,torch,json
sys.path[:0]=['/work/batch-full','/work/modelwide','/work/full-attention-native/prefix']
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from project_batch import project_body
from project_compress import project_body as single_project
from compressor_batch import compressor_body
from compressor import compressor_body as single_compress
from combined import sync
@g.jit
def batch(X,W,Y,C,COUNTER,HEALTH,META:gl.constexpr,N:gl.constexpr,M:gl.constexpr):
 p=gl.program_id(0)
 project_body(X,W,Y,N,M)
 sync(COUNTER,HEALTH,256,3)
 if p<M:
  CR=C[:4]+(C[4]+p*4,C[5]+p*4,C[6]+p,C[7]+p)+C[8:12]+(C[12]+p*16,)
  compressor_body(Y+p*N,Y+p*N+N//2,*CR,*META,OWNER=p)
@g.jit
def single(X,W,Y,C,COUNTER,HEALTH,META:gl.constexpr,N:gl.constexpr):
 single_project(X,W,Y,N)
 sync(COUNTER,HEALTH,256,3)
 single_compress(Y,Y+N//2,*C,*META)
def rand(*s):return torch.randn(*s,device='cuda')
def iz(*s):return torch.zeros(*s,device='cuda',dtype=torch.int32)
