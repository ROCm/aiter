from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

@g.jit
def local_quant(v,N:gl.constexpr,GROUP:gl.constexpr):
    block=gl.reshape(v,(N//GROUP,GROUP))
    amax=gl.maximum(gl.max(gl.abs(block),1),1.e-10)
    bits=(amax/448.).to(gl.uint32,bitcast=True)
    exponent=gl.minimum(((bits&0x7fffffff)+0x7fffff)>>23,254)
    scale=(exponent<<23).to(gl.float32,bitcast=True)
    q=(block/scale[:,None]).to(gl.float8e4nv).to(gl.float32)*scale[:,None]
    return gl.reshape(q,(N,))

@g.jit
def mhc_local(R,M,W,X,XQ):
    if gl.program_id(0)==0:
        i=gl.arange(0,4096,layout=gl.BlockedLayout([1],[32],[4],[0]))
        v=gl.full((4096,),0.,gl.float32,gl.BlockedLayout([1],[32],[4],[0]))
        for h in gl.static_range(4):
            v+=gl.load(R+h*4096+i).to(gl.float32)*gl.load(M+h)
        v=v.to(gl.bfloat16).to(gl.float32)
        v=v*gl.rsqrt(gl.sum(v*v,0)/4096+1.e-6)*gl.load(W+i).to(gl.float32)
        v=v.to(gl.bfloat16).to(gl.float32)
        gl.store(X+i,v)
        gl.store(XQ+i,local_quant(v,4096,128))

@g.jit
def act_quant_local(G,U,A,AQ,ROWS:gl.constexpr,GROUP:gl.constexpr,ROUND:gl.constexpr):
    row=gl.program_id(0)
    if row<ROWS:
        i=gl.arange(0,2048,layout=gl.BlockedLayout([1],[32],[4],[0]))
        gate=gl.minimum(gl.load(G+row*2048+i).to(gl.float32),10.)
        up=gl.minimum(gl.maximum(gl.load(U+row*2048+i).to(gl.float32),-10.),10.)
        v=gate/(1.+gl.exp(-gate))*up
        if ROUND:v=v.to(gl.bfloat16).to(gl.float32)
        gl.store(A+row*2048+i,v)
        gl.store(AQ+row*2048+i,local_quant(v,2048,GROUP))

@g.jit
def combine_post(EO,SO,R,M,Sum,Out):
    tile=gl.program_id(0)
    if tile<16:
        i=tile*256+gl.arange(0,256,layout=gl.BlockedLayout([1],[32],[4],[0]))
        v=gl.full((256,),0.,gl.float32,gl.BlockedLayout([1],[32],[4],[0]))
        for slot in gl.static_range(6):v+=gl.load(EO+slot*4096+i).to(gl.float32)
        v=(v.to(gl.bfloat16).to(gl.float32)+gl.load(SO+i).to(gl.float32)).to(gl.bfloat16).to(gl.float32)
        gl.store(Sum+i,v)
        for h in gl.static_range(4):
            y=v*gl.load(M+4+h)
            for source in gl.static_range(4):
                y+=gl.load(R+source*4096+i).to(gl.float32)*gl.load(M+8+source*4+h)
            gl.store(Out+h*4096+i,y.to(gl.bfloat16))
