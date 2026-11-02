import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_load,dev_store
from combined import sync
from tail import quant_scale

@g.jit
def inverse_quant(O,Cos,Sin,Q,S):
    pid=gl.program_id(0)
    L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
    i=gl.arange(0,128,layout=L)
    x=dev_load(O,pid*128+i).to(gl.float32)
    if pid%4==3:
        other=gl.gather(x,i^1,0)
        c=dev_load(Cos,(i-64)//2,i>=64,1.,cache='.ca')
        s=dev_load(Sin,(i-64)//2,i>=64,0.,cache='.ca')
        x=x*c+gl.where(i%2==0,1.,-1.)*other*s
    e,scale=quant_scale(x)
    dev_store((x/scale).to(gl.float8e4nv),Q,pid*128+i)
    dev_store(e,S,pid)

@g.jit
def split_project(X,XS,W,WS,P,N:gl.constexpr,K:gl.constexpr,SPLIT:gl.constexpr,GROUPED:gl.constexpr):
    pid=gl.program_id(0);nt=pid//SPLIT;ks=pid%SPLIT
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,128])
    DA:gl.constexpr=gl.DotOperandLayout(0,ML,16);DB:gl.constexpr=gl.DotOperandLayout(1,ML,16)
    ASL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DA,[16,4],scale_factor=32)
    BSL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DB,[128,4],scale_factor=32)
    AL:gl.constexpr=gl.to_linear_layout(DA,[16,128]);BL:gl.constexpr=gl.to_linear_layout(DB,[128,128])
    SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[128,128],[1,0])
    XL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[1,128],[1,0])
    sh=gl.allocate_shared_memory(W.dtype.element_ty,[2,128,128],SL)
    xsh=gl.allocate_shared_memory(X.dtype.element_ty,[2,1,128],XL)
    wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=W+nt*128*K,shape=[128,K],strides=[K,1],block_shape=[128,128],layout=SL)
    xb=nt//8*K if GROUPED else 0
    xd=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=X+xb,shape=[1,K],strides=[K,1],block_shape=[1,128],layout=XL)
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL));ka=gl.arange(0,128,layout=gl.SliceLayout(0,AL))
    kb=gl.arange(0,128,layout=gl.SliceLayout(1,BL));rn=gl.arange(0,128,layout=gl.SliceLayout(0,BL))
    blocks:gl.constexpr=K//SPLIT//128
    first=ks*blocks
    gl.amd.gfx1250.tdm.async_load(wd,[0,first*128],sh.index(0));gl.amd.gfx1250.tdm.async_load(xd,[0,first*128],xsh.index(0))
    gl.amd.gfx1250.tdm.async_load(wd,[0,(first+1)*128],sh.index(1));gl.amd.gfx1250.tdm.async_load(xd,[0,(first+1)*128],xsh.index(1))
    acc=gl.full((16,128),0,gl.float32,ML)
    for b in range(blocks):
        if b<blocks-1:gl.amd.gfx1250.tdm.async_wait(2)
        else:gl.amd.gfx1250.tdm.async_wait(0)
        bv=gl.amd.cdna4.async_copy.load_shared_relaxed(sh.index(b%2).permute([1,0]),BL)
        vx=gl.amd.cdna4.async_copy.load_shared_relaxed(xsh.index(b%2),gl.BlockedLayout([1,4],[1,32],[4,1],[1,0]))
        vx=gl.sum(vx.to(gl.float32),0)
        av=gl.convert_layout(vx,gl.SliceLayout(0,AL))[None,:]+gl.full((16,128),0,gl.float32,AL)
        gl.barrier()
        if b<blocks-2:
            gl.amd.gfx1250.tdm.async_load(wd,[0,(first+b+2)*128],sh.index(b%2));gl.amd.gfx1250.tdm.async_load(xd,[0,(first+b+2)*128],xsh.index(b%2))
        ax=gl.load(XS+xb//128+first+b).to(gl.uint32)
        bx=gl.load(WS+nt*(K//128)+first+b).to(gl.uint32)
        asc=ax.to(gl.uint8)+gl.full((16,4),0,gl.uint8,ASL)
        bsc=bx.to(gl.uint8)+gl.full((128,4),0,gl.uint8,BSL)
        aa=gl.convert_layout(av.to(gl.float8e4nv),DA);bb=gl.convert_layout(bv,DB)
        acc=gl.amd.gfx1250.wmma_scaled(aa,asc,'e4m3',bb,bsc,'e4m3',acc)
    OL:gl.constexpr=gl.to_linear_layout(ML,[16,128]);out=gl.convert_layout(acc,OL)
    om=gl.arange(0,16,layout=gl.SliceLayout(1,OL));on=nt*128+gl.arange(0,128,layout=gl.SliceLayout(0,OL))
    dev_store(out,P,ks*N+on[None,:]+om[:,None]*0,om[:,None]==0)

@g.jit
def reduce_quant(P,Y,Q,S,N:gl.constexpr,SPLIT:gl.constexpr,QUANT:gl.constexpr):
    pid=gl.program_id(0)
    if pid<N//128:
        i=gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
        x=gl.full((128,),0,gl.float32,gl.BlockedLayout([1],[32],[4],[0]))
        for k in gl.static_range(SPLIT):x+=dev_load(P,k*N+pid*128+i)
        x=x.to(gl.bfloat16).to(gl.float32)
        dev_store(x,Y,pid*128+i)
        if QUANT:
            e,s=quant_scale(x);dev_store((x/s).to(gl.float8e4nv),Q,pid*128+i);dev_store(e,S,pid)

@g.jit
def output_body(O,Cos,Sin,OA,OAS,OB,OBS,XQ,XS,AP,AY,AQ,AS,BP,Y,Counter,Health,SA:gl.constexpr=4,SB:gl.constexpr=8):
    inverse_quant(O,Cos,Sin,XQ,XS)
    sync(Counter,Health,256,3)
    if gl.program_id(0)<64*SA:split_project(XQ,XS,OA,OAS,AP,8192,4096,SA,True)
    sync(Counter,Health,256,3)
    reduce_quant(AP,AY,AQ,AS,8192,SA,True)
    sync(Counter,Health,256,3)
    if gl.program_id(0)<32*SB:split_project(AQ,AS,OB,OBS,BP,4096,8192,SB,False)
    sync(Counter,Health,256,3)
    reduce_quant(BP,Y,AQ,AS,4096,SB,False)

@g.jit
def output_kernel(O,Cos,Sin,OA,OAS,OB,OBS,XQ,XS,AP,AY,AQ,AS,BP,Y,Counter,Health,SA:gl.constexpr=4,SB:gl.constexpr=8):
    output_body(O,Cos,Sin,OA,OAS,OB,OBS,XQ,XS,AP,AY,AQ,AS,BP,Y,Counter,Health,SA,SB)

@g.jit
def output_stage(O,Cos,Sin,OA,OAS,OB,OBS,XQ,XS,AP,AY,AQ,AS,BP,Y,STAGE:gl.constexpr,SA:gl.constexpr=4,SB:gl.constexpr=8):
    if STAGE==0:inverse_quant(O,Cos,Sin,XQ,XS)
    elif STAGE==1:
        if gl.program_id(0)<64*SA:split_project(XQ,XS,OA,OAS,AP,8192,4096,SA,True)
    elif STAGE==2:reduce_quant(AP,AY,AQ,AS,8192,SA,True)
    elif STAGE==3:
        if gl.program_id(0)<32*SB:split_project(AQ,AS,OB,OBS,BP,4096,8192,SB,False)
    else:reduce_quant(BP,Y,AQ,AS,4096,SB,False)
