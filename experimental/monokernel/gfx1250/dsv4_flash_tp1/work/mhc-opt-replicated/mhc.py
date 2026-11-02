from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_load,dev_store
from combined import sync

@g.jit
def sinkhorn(M):
    if gl.program_id(0) == 0:
        L: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
        r = gl.arange(0, 4, layout=gl.SliceLayout(1, L))
        c = gl.arange(0, 4, layout=gl.SliceLayout(0, L))
        z = dev_load(M, 8 + r[:, None] * 4 + c[None, :], cache='.cg')
        z = gl.exp(z - gl.max(z, 1)[:, None])
        z = z / gl.sum(z, 1)[:, None] + 1e-06
        z = z / (gl.sum(z, 0)[None, :] + 1e-06)
        for _ in range(19):
            z = z / (gl.sum(z, 1)[:, None] + 1e-06)
            z = z / (gl.sum(z, 0)[None, :] + 1e-06)
        dev_store(z, M, 8 + r[:, None] * 4 + c[None, :], cache='.cg')
        v = dev_load(M, r, cache='.cg')
        dev_store(1.0 / (1.0 + gl.exp(-v)) + 1e-06, M, r, cache='.cg')
        v = dev_load(M, 4 + r, cache='.cg')
        dev_store(2.0 / (1.0 + gl.exp(-v)), M, 4 + r, cache='.cg')

@g.jit
def mhc_pre(R, M, Y, GRID: gl.constexpr):
    i = gl.program_id(0) * 256 + gl.arange(0, 256, layout=gl.BlockedLayout([1], [32], [4], [0]))
    for base in range(0, 4096, GRID * 256):
        c = base + i
        y = gl.full((256,), 0, gl.float32, gl.BlockedLayout([1], [32], [4], [0]))
        for h in gl.static_range(4):
            y += dev_load(R, h * 4096 + c, c < 4096, 0, cache='.cg').to(gl.float32) * dev_load(M, h, cache='.cg')
        dev_store(y.to(gl.bfloat16), Y, c, c < 4096, cache='.cg')

@g.jit
def norm(X, W, Y, N: gl.constexpr, ROWS: gl.constexpr, WEIGHT: gl.constexpr, GRID: gl.constexpr, ROUND: gl.constexpr=True):
    i = gl.arange(0, N, layout=gl.BlockedLayout([1], [32], [4], [0]))
    for row in range(gl.program_id(0), ROWS, GRID):
        x = dev_load(X, row * N + i, cache='.cg').to(gl.float32)
        y = x * gl.rsqrt(gl.sum(x * x, 0) / N + 1e-06)
        if WEIGHT:
            y *= dev_load(W, i, cache='.cg').to(gl.float32)
        if ROUND:
            y = y.to(gl.bfloat16)
        dev_store(y, Y, row * N + i, cache='.cg')


@g.jit
def mhc_project(R,W,Temp):
    pid=gl.program_id(0);nt=pid//128;ks=pid%128
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,32])
    AL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    BL:gl.constexpr=gl.BlockedLayout([4,1],[32,1],[1,4],[0,1])
    SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[16,128],[1,0])
    xs=gl.allocate_shared_memory(R.dtype.element_ty,[16,128],SL)
    ws=gl.allocate_shared_memory(W.dtype.element_ty,[16,128],SL)
    xd=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=R,shape=[1,16384],strides=[16384,1],block_shape=[16,128],layout=SL)
    wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=W,shape=[24,16384],strides=[16384,1],block_shape=[16,128],layout=SL)
    gl.amd.gfx1250.tdm.async_load(xd,[0,ks*128],xs,cache_modifier='.cg')
    gl.amd.gfx1250.tdm.async_load(wd,[nt*16,ks*128],ws,cache_modifier='.cg')
    gl.amd.gfx1250.tdm.async_wait(0)
    av=xs.load(AL);w=ws.permute([1,0]).load(BL)
    hi=w.to(gl.bfloat16);lo=(w-hi.to(gl.float32)).to(gl.bfloat16)
    aa=gl.convert_layout(av,gl.DotOperandLayout(0,ML,8))
    acc=gl.amd.gfx1250.wmma(aa,gl.convert_layout(hi,gl.DotOperandLayout(1,ML,8)),gl.full((16,16),0,gl.float32,ML))
    acc=gl.amd.gfx1250.wmma(aa,gl.convert_layout(lo,gl.DotOperandLayout(1,ML,8)),acc)
    out=gl.convert_layout(acc,AL)
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL));n=nt*16+gl.arange(0,16,layout=gl.SliceLayout(0,AL))
    dev_store(out,Temp,ks*24+n[None,:]+rm[:,None]*0,(rm[:,None]==0)&(n[None,:]<24))

@g.jit
def mhc_reduce(R,S,B,M,Temp):
    row=gl.program_id(0)
    if row<24:
        L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
        i=gl.arange(0,1024,layout=L);si=gl.arange(0,128,layout=L)
        dot=gl.sum(dev_load(Temp,si*24+row),0);sq=0.
        for b in range(16):
            x=dev_load(R,b*1024+i).to(gl.float32);sq+=gl.sum(x*x,0)
        z=dot*gl.rsqrt(sq/16384+1.e-6);idx=gl.where(row<4,0,gl.where(row<8,1,2))
        dev_store(z*dev_load(S,idx)+dev_load(B,row),M,row)

@g.jit
def mhc_stage(R,W,S,B,AN,M,Temp,Raw,X,STAGE:gl.constexpr):
    if STAGE==0:mhc_project(R,W,Temp)
    elif STAGE==1:mhc_reduce(R,S,B,M,Temp)
    elif STAGE==2:sinkhorn(M)
    elif STAGE==3:mhc_pre(R,M,Raw,16)
    elif STAGE==4:norm(Raw,AN,X,4096,1,True,1)

@g.jit
def mhc_body(R,W,S,B,AN,M,Temp,Raw,X,Counter,Health):
    mhc_project(R,W,Temp)
    sync(Counter,Health,256,3)
    mhc_reduce(R,S,B,M,Temp)
    sync(Counter,Health,256,3)
    sinkhorn(M)
    sync(Counter,Health,256,3)
    mhc_pre(R,M,Raw,256)
    sync(Counter,Health,256,3)
    norm(Raw,AN,X,4096,1,True,256)

@g.jit
def mhc_region(R,W,S,B,AN,M,Temp,Raw,X,Counter,Health):
    mhc_body(R,W,S,B,AN,M,Temp,Raw,X,Counter,Health)
