from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def compare(A,B,Stats,N:gl.constexpr,SLOT:gl.constexpr,BLOCK:gl.constexpr=1024):
 p=gl.program_id(0);i=p*BLOCK+gl.arange(0,BLOCK,layout=gl.BlockedLayout([1],[32],[4],[0]));ok=i<N
 a=gl.load(A+i,ok,0.).to(gl.float32);b=gl.load(B+i,ok,0.).to(gl.float32)
 finite=(gl.abs(a)<float('inf'))&(gl.abs(b)<float('inf'))
 n=gl.sum((~finite&ok).to(gl.int32),0)
 d=gl.where(finite,gl.abs(a-b),0)
 gl.atomic_add(Stats+SLOT*3,n.to(gl.float32),sem='relaxed')
 gl.atomic_max(Stats+SLOT*3+1,gl.max(d,0),sem='relaxed')
 gl.atomic_max(Stats+SLOT*3+2,gl.max(gl.where(ok,gl.abs(b),0),0),sem='relaxed')
