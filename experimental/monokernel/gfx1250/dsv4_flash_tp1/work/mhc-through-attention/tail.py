import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_load,dev_store
@g.jit
def quant_scale(x):
    a=gl.maximum(gl.max(gl.abs(x),0),1.e-10)/448.
    e=gl.minimum(((a.to(gl.uint32,bitcast=True)&2147483647)+8388607)>>23,254)
    return e.to(gl.uint8),(e<<23).to(gl.float32,bitcast=True)
@g.jit
def quant(X,Q,S):
    i=gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
    b=gl.program_id(0)
    x=dev_load(X,b*128+i,cache='.ca').to(gl.float32)
    e,s=quant_scale(x)
    dev_store((x/s).to(gl.float8e4nv),Q,b*128+i)
    dev_store(e,S,b)
@g.jit
def norm_rope_body(Q,Cos,Sin,O,HEAD,DO_NORM:gl.constexpr,WAVES:gl.constexpr=4):
    i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[WAVES],[0]))
    x=dev_load(Q,HEAD*512+i).to(gl.float32)
    if DO_NORM:x=x*gl.rsqrt(gl.sum(x*x,0)/512+1.e-6)
    other=gl.gather(x,i^1,0)
    c=dev_load(Cos,(i-448)//2,i>=448,1.,cache='.ca')
    s=dev_load(Sin,(i-448)//2,i>=448,0.,cache='.ca')
    dev_store(x*c+gl.where(i%2==0,-1.,1.)*other*s,O,HEAD*512+i)
@g.jit
def norm_rope(Q,Cos,Sin,O,DO_NORM:gl.constexpr):
    norm_rope_body(Q,Cos,Sin,O,gl.program_id(0),DO_NORM)
@g.jit
def norm_only(Q,O):
    i=gl.arange(0,512,layout=gl.BlockedLayout([1],[32],[4],[0]))
    h=gl.program_id(0);x=dev_load(Q,h*512+i).to(gl.float32)
    dev_store(x*gl.rsqrt(gl.sum(x*x,0)/512+1.e-6),O,h*512+i)
