import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_load,dev_store

@g.jit
def drain():
    gl.inline_asm_elementwise('s_wait_loadcnt 0\n s_wait_storecnt 0\n s_mov_b32 $0, 0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
    gl.barrier()

@g.jit
def initialize(Acc,Count):
    i=gl.program_id(0)*256+gl.arange(0,256,layout=gl.BlockedLayout([1],[32],[4],[0]))
    dev_store(0.,Acc,i,i<1536)
    dev_store(0,Count,i,i<50)

@g.jit
def finish_group(Acc,SS,NW,Y,BASE:gl.constexpr,N:gl.constexpr):
    L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
    t=gl.arange(0,N//32,layout=L)
    total=gl.sum(dev_load(SS,BASE//32+t),0)
    i=gl.arange(0,N,layout=L)
    x=dev_load(Acc,BASE+i)
    y=x*gl.rsqrt(total/N+1.e-6)*dev_load(NW,BASE+i)
    dev_store(y,Y,BASE+i)

@g.jit
def project(X,XS,W,WS,NW,Part,Acc,SS,Count,Y,MODE:gl.constexpr):
    # MODE0=A scratch;1=B atomic fused;2=C atomic standalone;3=D scratch fused.
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,32])
    AL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    BL:gl.constexpr=gl.BlockedLayout([4,1],[32,1],[1,4],[0,1])
    DA:gl.constexpr=gl.DotOperandLayout(0,ML,8)
    DB:gl.constexpr=gl.DotOperandLayout(1,ML,8)
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
    ka=gl.arange(0,128,layout=gl.SliceLayout(0,AL))
    kb=gl.arange(0,128,layout=gl.SliceLayout(1,BL))
    rn=gl.arange(0,32,layout=gl.SliceLayout(0,BL))
    tile=gl.program_id(0)//8;ks=gl.program_id(0)%8
    ns=tile*32+rn
    acc=gl.full((16,32),0,gl.float32,ML)
    for block in range(ks*4,(ks+1)*4):
        av=dev_load(X,block*128+ka[None,:]+rm[:,None]*0,rm[:,None]==0,0.)
        bv=dev_load(W,ns[None,:]*4096+block*128+kb[:,None])
        ax=dev_load(XS,block).to(gl.uint32)
        ax=(ax<<23).to(gl.float32,bitcast=True)
        bx=dev_load(WS,(ns//128)*32+block).to(gl.uint32)
        bx=(bx<<23).to(gl.float32,bitcast=True)
        aa=gl.convert_layout((av.to(gl.float32)*ax).to(gl.bfloat16),DA)
        bb=gl.convert_layout((bv.to(gl.float32)*bx[None,:]).to(gl.bfloat16),DB)
        acc=gl.amd.gfx1250.wmma(aa,bb,acc)
    out=gl.convert_layout(acc,AL)
    om=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
    on=tile*32+gl.arange(0,32,layout=gl.SliceLayout(0,AL))
    if MODE==0 or MODE==3:
        dev_store(out,Part,ks*1536+on[None,:]+om[:,None]*0,om[:,None]==0)
    else:
        gl.atomic_add(Acc+on[None,:]+om[:,None]*0,out,sem='relaxed',scope='gpu',mask=om[:,None]==0)
    if MODE==1 or MODE==3:
        drain()
        ticket=gl.atomic_add(Count+tile,1,sem='acq_rel',scope='gpu')
        if ticket==7:
            i=gl.arange(0,32,layout=gl.BlockedLayout([1],[32],[4],[0]))
            if MODE==3:
                x=gl.full((32,),0.,gl.float32,gl.BlockedLayout([1],[32],[4],[0]))
                for s in gl.static_range(8):x+=dev_load(Part,s*1536+tile*32+i)
            else:x=dev_load(Acc,tile*32+i)
            x=x.to(gl.bfloat16).to(gl.float32)
            dev_store(x,Acc,tile*32+i)
            ss=gl.sum(x*x,0)
            dev_store(ss,SS,tile)
            drain()
            group=gl.where(tile<32,0,1)
            done=gl.atomic_add(Count+48+group,1,sem='acq_rel',scope='gpu')
            if done==gl.where(tile<32,31,15):
                if tile<32:finish_group(Acc,SS,NW,Y,0,1024)
                else:finish_group(Acc,SS,NW,Y,1024,512)

@g.jit
def reduce_group(Part,Acc,NW,Y,BASE:gl.constexpr,N:gl.constexpr,ATOMIC:gl.constexpr):
    i=gl.arange(0,N,layout=gl.BlockedLayout([1],[32],[4],[0]))
    if ATOMIC:x=dev_load(Acc,BASE+i)
    else:
        x=gl.full((N,),0.,gl.float32,gl.BlockedLayout([1],[32],[4],[0]))
        for s in gl.static_range(8):x+=dev_load(Part,s*1536+BASE+i)
    x=x.to(gl.bfloat16).to(gl.float32)
    y=x*gl.rsqrt(gl.sum(x*x,0)/N+1.e-6)*dev_load(NW,BASE+i)
    dev_store(y,Y,BASE+i)

@g.jit
def reduce_norm(Part,Acc,NW,Y,ATOMIC:gl.constexpr):
    if gl.program_id(0)==0:reduce_group(Part,Acc,NW,Y,0,1024,ATOMIC)
    else:reduce_group(Part,Acc,NW,Y,1024,512,ATOMIC)
