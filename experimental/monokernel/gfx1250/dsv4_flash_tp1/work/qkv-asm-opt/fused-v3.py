import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_load,dev_store

@g.jit
def project_fused(X,XS,W,WS,P,NW,Acc,Count,Y,MODE:gl.constexpr,BN:gl.constexpr,SPLIT:gl.constexpr,DIRECT:gl.constexpr,WAVES:gl.constexpr,BK:gl.constexpr,UNROLL:gl.constexpr,UPSCALE:gl.constexpr=False,REPLICATE:gl.constexpr=False,CACHE:gl.constexpr=".cg",SCALAR:gl.constexpr=False,PACKED:gl.constexpr=False,OUTLAY:gl.constexpr=0,NVEC:gl.constexpr=1):
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
        if REPLICATE:av=dev_load(X,block*BK+ka[None,:]+rm[:,None]*0,cache=CACHE)
        else:av=dev_load(X,block*BK+ka[None,:]+rm[:,None]*0,rm[:,None]==0,0.,cache=CACHE)
        if PACKED:
            kk=block*BK+kb[:,None]
            wi=((ns[None,:]//16)*128+kk//32)*512+(ns[None,:]%16)*32+kk%32
        else:wi=ns[None,:]*4096+block*BK+kb[:,None]
        bv=dev_load(W,wi,cache=CACHE)
        if SCALAR and BK==128:
            ax=gl.load(XS+block)+gl.full((16,BK),0,gl.uint8,AL)
            bx=gl.load(WS+(tile*BN//128)*32+block)+gl.full((BK,BN),0,gl.uint8,BL)
        else:
            ax=dev_load(XS,(block*BK+ka[None,:])//128,cache=CACHE)
            bx=dev_load(WS,(ns[None,:]//128)*32+(block*BK+kb[:,None])//128,cache=CACHE)
        if UPSCALE:
            a16=gl.amd.gfx1250.scaled_upcast(av,ax+gl.full((16,BK),0,gl.uint8,AL),gl.bfloat16)
            if W.dtype.element_ty==gl.bfloat16:b16=bv
            else:b16=gl.amd.gfx1250.scaled_upcast(bv,bx,gl.bfloat16)
        else:
            ax=(ax.to(gl.uint32)<<23).to(gl.float32,bitcast=True)
            bx=(bx.to(gl.uint32)<<23).to(gl.float32,bitcast=True)
            a16=(av.to(gl.float32)*ax).to(gl.bfloat16)
            b16=(bv.to(gl.float32)*bx).to(gl.bfloat16)
        aa=gl.convert_layout(a16,DA)
        bb=gl.convert_layout(b16,DB)
        acc=gl.amd.gfx1250.wmma(aa,bb,acc)
    OL:gl.constexpr=gl.to_linear_layout(ML,[16,BN]) if OUTLAY==0 else gl.BlockedLayout([1,1],[2,16],[1,WAVES],[1,0]) if OUTLAY==1 else gl.BlockedLayout([1,1],[1,32],[1,WAVES],[1,0])
    out=gl.convert_layout(acc,OL)
    om=gl.arange(0,16,layout=gl.SliceLayout(1,OL))
    on=tile*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,OL))
    if MODE==0:dev_store(out,P,ks*1536+on[None,:]+om[:,None]*0,om[:,None]==0)
    else:gl.atomic_add(Acc+on[None,:]+om[:,None]*0,out,sem='relaxed',scope='gpu',mask=om[:,None]==0)
    gl.inline_asm_elementwise('s_wait_loadcnt 0\n s_wait_storecnt 0\n s_mov_b32 $0, 0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
    if WAVES>1:gl.barrier()
    group=gl.where(tile*BN<1024,0,1)
    ticket=gl.atomic_add(Count+group,1,sem='relaxed',scope='gpu')
    if ticket==gl.where(group==0,1024//BN*SPLIT-1,512//BN*SPLIT-1):
        if group==0:finish_fused(P,Acc,NW,Count,Y,0,1024,SPLIT,WAVES,MODE,NVEC)
        else:finish_fused(P,Acc,NW,Count,Y,1024,512,SPLIT,WAVES,MODE,NVEC)

@g.jit
def finish_fused(P,Acc,NW,Count,Y,BASE:gl.constexpr,N:gl.constexpr,SPLIT:gl.constexpr,WAVES:gl.constexpr,MODE:gl.constexpr,NVEC:gl.constexpr):
    L:gl.constexpr=gl.BlockedLayout([NVEC],[32],[WAVES],[0])
    i=gl.arange(0,N,layout=L)
    if MODE==0:
        x=gl.full((N,),0.,gl.float32,L)
        for s in gl.static_range(SPLIT):x+=dev_load(P,s*1536+BASE+i)
    else:x=dev_load(Acc,BASE+i)
    x=x.to(gl.bfloat16).to(gl.float32)
    y=x*gl.rsqrt(gl.sum(x*x,0)/N+1.e-6)*dev_load(NW,BASE+i)
    dev_store(y,Y,BASE+i)
    if MODE==1:dev_store(0.,Acc,BASE+i)
    gl.atomic_xchg(Count+BASE//1024,0,sem='relaxed',scope='gpu')
