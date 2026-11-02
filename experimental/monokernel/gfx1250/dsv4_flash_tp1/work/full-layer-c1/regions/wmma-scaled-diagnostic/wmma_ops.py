from kernel import *
@g.jit
def pack(X,Y,P,S,N:gl.constexpr,GRID:gl.constexpr):
    i=gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
    for group in range(gl.program_id(0),N//128,GRID):
        x=dev_load(X,group*128+i,cache='.cg').to(gl.float32)
        amax=gl.maximum(gl.max(gl.abs(x),0),1e-10)
        bits=(amax/448.).to(gl.uint32,bitcast=True)
        exponent=gl.minimum(((bits&2147483647)+8388607)>>23,254)
        scale=(exponent<<23).to(gl.float32,bitcast=True)
        q=(x/scale).to(gl.float8e4nv)
        dev_store(q,P,group*128+i,cache='.cg')
        dev_store(exponent.to(gl.uint8),S,group,cache='.cg')
        dev_store(q.to(gl.float32)*scale,Y,group*128+i,cache='.cg')
@g.jit
def wmma_matvec(X,XS,W,S,Y,N:gl.constexpr,K:gl.constexpr,GRID:gl.constexpr,BN:gl.constexpr=32,SPLIT:gl.constexpr=1):
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,128])
    AL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    BL:gl.constexpr=gl.BlockedLayout([4,1],[32,1],[1,4],[0,1])
    CL:gl.constexpr=AL
    DA:gl.constexpr=gl.DotOperandLayout(0,ML,16)
    DB:gl.constexpr=gl.DotOperandLayout(1,ML,16)
    SA:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DA,[16,4],scale_factor=32)
    SB:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DB,[BN,4],scale_factor=32)
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
    ka=gl.arange(0,128,layout=gl.SliceLayout(0,AL))
    kb=gl.arange(0,128,layout=gl.SliceLayout(1,BL))
    rn=gl.arange(0,BN,layout=gl.SliceLayout(0,BL))
    sm=gl.arange(0,16,layout=gl.SliceLayout(1,SA))
    sk=gl.arange(0,4,layout=gl.SliceLayout(0,SA))
    sn=gl.arange(0,BN,layout=gl.SliceLayout(1,SB))
    sbk=gl.arange(0,4,layout=gl.SliceLayout(0,SB))
    for tile in range(gl.program_id(0),gl.cdiv(N,BN)*SPLIT,GRID):
        ntile=tile//SPLIT
        ks=tile%SPLIT
        ns=ntile*BN+rn
        acc=gl.full((16,BN),0,gl.float32,ML)
        for kblock in range(ks*(K//128//SPLIT),(ks+1)*(K//128//SPLIT)):
            av=dev_load(X,kblock*128+ka[None,:]+rm[:,None]*0,rm[:,None]==0,0.0,cache='.cg')
            bv=dev_load(W,ns[None,:]*K+kblock*128+kb[:,None],cache='.cg')
            aa=gl.convert_layout(av,DA);bb=gl.convert_layout(bv,DB)
            asc=dev_load(XS,kblock+sm[:,None]*0+sk[None,:]*0,cache='.cg')
            bsc=dev_load(S,((ntile*BN+sn[:,None])//128)*(K//128)+kblock+sbk[None,:]*0,cache='.cg')
            acc=gl.amd.gfx1250.wmma_scaled(aa,asc,'e4m3',bb,bsc,'e4m3',acc)
        out=gl.convert_layout(acc,CL)
        om=gl.arange(0,16,layout=gl.SliceLayout(1,CL))
        on=ntile*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,CL))
        if SPLIT == 1:
            out=out.to(gl.bfloat16)
        dev_store(out,Y,ks*N+on[None,:]+om[:,None]*0,om[:,None]==0,cache='.cg')

@g.jit
def partial_norm(P,W,Y,N:gl.constexpr,SPLIT:gl.constexpr):
    if gl.program_id(0)==0:
        i=gl.arange(0,N,layout=gl.BlockedLayout([1],[32],[4],[0]))
        x=gl.full((N,),0,gl.float32,gl.BlockedLayout([1],[32],[4],[0]))
        for s in gl.static_range(SPLIT):
            x+=dev_load(P,s*N+i,cache='.cg')
        x=x.to(gl.bfloat16).to(gl.float32)
        y=x*gl.rsqrt(gl.sum(x*x,0)/N+1e-6)*dev_load(W,i,cache='.cg').to(gl.float32)
        dev_store(y,Y,i,cache='.cg')

@g.jit
def projection_probe(X,XS,W,S,Y,N:gl.constexpr,K:gl.constexpr,GRID:gl.constexpr):
    wmma_matvec(X,XS,W,S,Y,N,K,GRID,128)
