import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_load,dev_store

@g.jit
def project_opt(X,XS,W,WS,P,BN:gl.constexpr,SPLIT:gl.constexpr,DIRECT:gl.constexpr,WAVES:gl.constexpr,BK:gl.constexpr,UNROLL:gl.constexpr,UPSCALE:gl.constexpr=False,REPLICATE:gl.constexpr=False):
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]] if WAVES==4 else [[0,1]] if WAVES==2 else [],instr_shape=[16,16,32])
    DA:gl.constexpr=gl.DotOperandLayout(0,ML,8)
    DB:gl.constexpr=gl.DotOperandLayout(1,ML,8)
    AL:gl.constexpr=gl.to_linear_layout(DA,[16,BK]) if DIRECT else gl.BlockedLayout([1,4],[1,32],[WAVES,1],[1,0])
    BL:gl.constexpr=gl.to_linear_layout(DB,[BK,BN]) if DIRECT else gl.BlockedLayout([4,1],[32,1],[1,WAVES],[0,1])
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
    ka=gl.arange(0,BK,layout=gl.SliceLayout(0,AL))
    kb=gl.arange(0,BK,layout=gl.SliceLayout(1,BL))
    rn=gl.arange(0,BN,layout=gl.SliceLayout(0,BL))
    tile=gl.program_id(0)//SPLIT;ks=gl.program_id(0)%SPLIT
    ns=tile*BN+rn
    acc=gl.full((16,BN),0,gl.float32,ML)
    for block in range(ks*(4096//SPLIT//BK),(ks+1)*(4096//SPLIT//BK)):
        if REPLICATE:av=dev_load(X,block*BK+ka[None,:]+rm[:,None]*0)
        else:av=dev_load(X,block*BK+ka[None,:]+rm[:,None]*0,rm[:,None]==0,0.)
        bv=dev_load(W,ns[None,:]*4096+block*BK+kb[:,None])
        ax=dev_load(XS,(block*BK+ka[None,:])//128)
        bx=dev_load(WS,(ns[None,:]//128)*32+(block*BK+kb[:,None])//128)
        if UPSCALE:
            a16=gl.amd.gfx1250.scaled_upcast(av,ax+gl.full((16,BK),0,gl.uint8,AL),gl.bfloat16)
            b16=gl.amd.gfx1250.scaled_upcast(bv,bx,gl.bfloat16)
        else:
            ax=(ax.to(gl.uint32)<<23).to(gl.float32,bitcast=True)
            bx=(bx.to(gl.uint32)<<23).to(gl.float32,bitcast=True)
            a16=(av.to(gl.float32)*ax).to(gl.bfloat16)
            b16=(bv.to(gl.float32)*bx).to(gl.bfloat16)
        aa=gl.convert_layout(a16,DA)
        bb=gl.convert_layout(b16,DB)
        acc=gl.amd.gfx1250.wmma(aa,bb,acc)
    OL:gl.constexpr=gl.to_linear_layout(ML,[16,BN])
    out=gl.convert_layout(acc,OL)
    om=gl.arange(0,16,layout=gl.SliceLayout(1,OL))
    on=tile*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,OL))
    dev_store(out,P,ks*1536+on[None,:]+om[:,None]*0,om[:,None]==0)

@g.jit
def rg(P,NW,Y,BASE:gl.constexpr,N:gl.constexpr,SPLIT:gl.constexpr):
    i=gl.arange(0,N,layout=gl.BlockedLayout([1],[32],[4],[0]))
    x=gl.full((N,),0.,gl.float32,gl.BlockedLayout([1],[32],[4],[0]))
    for s in gl.static_range(SPLIT):x+=dev_load(P,s*1536+BASE+i)
    x=x.to(gl.bfloat16).to(gl.float32)
    y=x*gl.rsqrt(gl.sum(x*x,0)/N+1.e-6)*dev_load(NW,BASE+i)
    dev_store(y,Y,BASE+i)

@g.jit
def reduce_opt(P,NW,Y,SPLIT:gl.constexpr):
    if gl.program_id(0)==0:rg(P,NW,Y,0,1024,SPLIT)
    else:rg(P,NW,Y,1024,512,SPLIT)
