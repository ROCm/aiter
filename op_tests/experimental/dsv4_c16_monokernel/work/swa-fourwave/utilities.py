import triton as tr
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

@g.jit
def unshuffle(X,Y):
 p=gl.program_id(0);i=p*1024+gl.arange(0,1024,layout=gl.BlockedLayout([1],[32],[4],[0]))
 row=i//16384;h=i//4096%4;k=i%4096
 src=(k//64*4+h)*16*64+row*64+k%64
 gl.store(Y+i,gl.load(X+src))

@g.jit
def act(X,Y):
 t=gl.program_id(0);k=gl.arange(0,2048,layout=gl.BlockedLayout([1],[32],[4],[0]))
 a=gl.minimum(gl.load(X+t*4096+k).to(gl.float32),10.)
 b=gl.minimum(gl.maximum(gl.load(X+t*4096+2048+k).to(gl.float32),-10.),10.)
 gl.store(Y+t*2048+k,a/(1.+gl.exp(-a))*b)

@g.jit
def clear(P,H,R):
 k=gl.arange(0,256,layout=gl.BlockedLayout([1],[32],[4],[0]))
 gl.store(P+k,0);gl.store(H+k,0);gl.store(R+k,0)

@g.jit
def add(A,B,C):
 i=gl.program_id(0)*256+gl.arange(0,256,layout=gl.BlockedLayout([1],[32],[4],[0]))
 gl.store(C+i,gl.load(A+i).to(gl.float32)+gl.load(B+i).to(gl.float32))

@g.jit
def add_row(A,B,C):
 i=gl.program_id(0)*4096+gl.arange(0,4096,layout=gl.BlockedLayout([1],[32],[4],[0]))
 gl.store(C+i,gl.load(A+i).to(gl.float32)+gl.load(B+i).to(gl.float32))

@g.jit
def add_native_vector_rows(A,B,C):
 # Native TensorIterator BF16 add: vector8, tile2048,256 logical threads.
 # One resident CTA owns a row =2 native tiles; four physical waves execute
 # both halves of each native tile. There is no cross-thread communication.
 row=gl.program_id(0)
 for tile in gl.static_range(2):
  i=row*4096+tile*2048+gl.arange(0,2048,layout=gl.BlockedLayout([8],[32],[4],[0]))
  gl.store(C+i,gl.load(A+i).to(gl.float32)+gl.load(B+i).to(gl.float32))
